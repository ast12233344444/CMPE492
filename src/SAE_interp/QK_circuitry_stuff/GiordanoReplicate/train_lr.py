import json
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchvision import datasets
from tqdm import tqdm
from nnsight import NNsight

from transformers import ViTForImageClassification, ViTImageProcessor

from CMPE492.src.SAE.train_sae import SparseAutoencoder
from CMPE492.src.SAE_interp.QK_circuitry_stuff.full_empatt_save import get_stratified_subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels

# --- 1. Disk-Backed Dataset for Regression ---
class BufferedRegressionDataset(IterableDataset):
    def __init__(self, cache_dir):
        """
        Streams chunked SAE features from disk, applies filtering masks,
        and normalizes by the global max values.
        """
        self.files = sorted([os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pt')])

    def __iter__(self):
        for file_path in self.files:
            data = torch.load(file_path, map_location="cpu")

            feats_src = data['src']
            feats_dest = data['dest']

            # Shuffle locally within the chunk
            indices = torch.randperm(feats_src.size(0))

            for idx in indices:
                yield feats_src[idx], feats_dest[idx]


# --- 2. Caching & Stat Tracking ---
def cache_all_tokens_and_stats(model, dataloader, SAE_src, SAE_dest, layer_src_idx, layer_dest_idx, cache_dir,
                               num_features):
    """
    Traces the post-LN/pre-attention latents using nnsight for ALL tokens,
    passes them through the SAEs, tracks normalization stats, and saves chunks.
    """
    os.makedirs(cache_dir, exist_ok=True)
    model.eval()
    SAE_src.eval()
    SAE_dest.eval()

    # Global trackers for normalization
    global_max_src = torch.zeros(num_features, device=device)
    global_max_dest = torch.zeros(num_features, device=device)

    global_counts_src = torch.zeros(num_features, device=device)
    global_counts_dest = torch.zeros(num_features, device=device)

    total_tokens = 0
    chunk_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Caching All Tokens L{layer_src_idx} -> L{layer_dest_idx}"):
            x_batch, _ = batch
            x_batch = x_batch.to(device)

            with model.trace() as tracer:
                with tracer.invoke(x_batch):
                    # Hook into the exact post-LN / pre-attention spot
                    act_src = model.vit.encoder.layer[layer_src_idx].layernorm_before.output
                    act_dest = model.vit.encoder.layer[layer_dest_idx].layernorm_before.output

                    encoded_src, _ = SAE_src(act_src)
                    encoded_dest, _ = SAE_dest(act_dest)

                    # Save the trace out of nnsight
                    encoded_src = encoded_src.save()
                    encoded_dest = encoded_dest.save()

            # Flatten: [Batch, 197, num_features] -> [Batch * 197, num_features]
            feats_src = encoded_src.value.view(-1, num_features)
            feats_dest = encoded_dest.value.view(-1, num_features)

            current_tokens = feats_src.size(0)
            total_tokens += current_tokens

            # Update running max
            global_max_src = torch.maximum(global_max_src, feats_src.max(dim=0).values)
            global_max_dest = torch.maximum(global_max_dest, feats_dest.max(dim=0).values)

            # Update running counts (L0)
            global_counts_src += (feats_src > 0).float().sum(dim=0)
            global_counts_dest += (feats_dest > 0).float().sum(dim=0)

            # Save chunk to disk (keep it on CPU to save VRAM)
            chunk_path = os.path.join(cache_dir, f"chunk_{chunk_idx:04d}.pt")
            torch.save({
                'src': feats_src.cpu(),
                'dest': feats_dest.cpu()
            }, chunk_path)

            chunk_idx += 1

            del feats_src, feats_dest, encoded_src, encoded_dest
            torch.cuda.empty_cache()

    # Calculate boolean masks for features that fire at least 1 in 1000 tokens
    threshold = 0
    mask_src = global_counts_src >= threshold
    mask_dest = global_counts_dest >= threshold

    #return mask_src, mask_dest, global_max_src, global_max_dest


# --- 3. Regression Training ---
def train_token_concept_graph(cache_dir, num_features, epochs=10, lr=1e-3, batch_size=4096):
    """
    Trains the linear regression mapping L -> L+1 using the chunked dataset.
    """
    dataset = BufferedRegressionDataset(cache_dir)
    loader = DataLoader(dataset, batch_size=batch_size)

    num_src_feats = num_features
    num_dest_feats = num_features

    print(f"Training Regression: {num_src_feats} Active Source Feats -> {num_dest_feats} Active Dest Feats")

    regression_model = nn.Linear(num_src_feats, num_dest_feats, bias=False).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(regression_model.parameters(), lr=lr)

    regression_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for src_batch, dest_batch in pbar:
            src_batch = src_batch.to(device)
            dest_batch = dest_batch.to(device)

            optimizer.zero_grad()
            predictions = regression_model(src_batch)
            loss = criterion(predictions, dest_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_count % 100 == 0:
                pbar.set_postfix({"mse": f"{loss.item():.6f}"})

        print(f"Epoch {epoch + 1} Average MSE: {total_loss / batch_count:.6f}")

    return regression_model.weight.detach().cpu()


# --- 4. Main Execution Wrapper ---
def build_and_save_token_graph(hf_model, dataloader, SAEs, cache_root, layer_pairs, num_features):
    # Ensure NNsight wrap
    if not hasattr(hf_model, 'trace'):
        model = NNsight(hf_model)
    else:
        model = hf_model

    for l_src, l_dest in layer_pairs:
        print(f"\n--- Processing Tokens: Layer {l_src} to Layer {l_dest} ---")

        chunk_cache_dir = os.path.join(cache_root, f"chunks_{l_src}_to_{l_dest}")
        SAE_src = SAEs[f"lnb{l_src}"]
        SAE_dest = SAEs[f"lnb{l_dest}"]

        try:
            # Step 1: Trace, extract all tokens, calculate normalizations, and chunk to disk
            cache_all_tokens_and_stats(
                model, dataloader, SAE_src, SAE_dest, l_src, l_dest, chunk_cache_dir, num_features
            )

            # Step 2: Stream from disk and train the N x N mapping
            weight_matrix = train_token_concept_graph(
                chunk_cache_dir, num_features
            )

            # Step 3: Save the final weight matrix and masks
            out_path = os.path.join(cache_root, f"token_concept_graph_L{l_src}_to_L{l_dest}.pt")
            torch.save({
                "weight_matrix": weight_matrix,
            }, out_path)
            print(f"Saved completed token-level graph to {out_path}")

        finally:
            # Step 4: Explicitly wipe the chunk cache from the drive
            # (Executes even if an error occurs during caching or training)
            if os.path.exists(chunk_cache_dir):
                shutil.rmtree(chunk_cache_dir)
                print(f"Cleaned up temporary cache directory: {chunk_cache_dir}")

# Example execution hook:
# build_and_save_graph(hf_model, dataloader, SAEs_dict, "results/concept_graphs", [(10, 11), (9, 10)])
if __name__ == "__main__":

    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager").to(device)
    processor = ViTImageProcessor.from_pretrained(model_id)
    model.eval()
    batch_size = 16

    dataset_full = datasets.CIFAR10(root='..\\..\\data', train=True, download=True)
    subset = get_stratified_subset(dataset_full, num_samples_per_class=500)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    SAE_dir = "C:\\Users\\ast12\\PycharmProjects\\CMPE492\\saved_models"
    SAE_act_stats_dir = f"C:\\Users\\ast12\\PycharmProjects\\CMPE492\\model_activation_stats"
    os.makedirs(SAE_act_stats_dir, exist_ok=True)
    saved_SAE_act_stats = os.listdir(SAE_act_stats_dir)
    SAEs = {}
    SAE_act_stats = {}
    for saved_SAE_act_stat in saved_SAE_act_stats:
        if "lnb" in saved_SAE_act_stat:
            layer_name = saved_SAE_act_stat.split("_")[1]
            model_name_corr = saved_SAE_act_stat.replace(".json", ".pt")
            SAE_act_stat_json = json.load(open(os.path.join(SAE_act_stats_dir, saved_SAE_act_stat)))
            SAE_act_stats[layer_name] = SAE_act_stat_json

            SAE_metadata = torch.load(os.path.join(SAE_dir, model_name_corr))
            SAE_model = SparseAutoencoder(expansion_factor=SAE_metadata['expansion_factor'])
            SAE_model.load_state_dict(SAE_metadata["model_state_dict"])
            SAE_model = SAE_model.to(device)
            SAEs[layer_name] = SAE_model

    NUM_FEATURES = int(768 * 16)
    cache_dir = "C:\\Users\\ast12\\PycharmProjects\\CMPE492\\results\\linreg_layer"

    layer_pairs = [(i, i+1) for i in range(6, 11)]
    build_and_save_token_graph(model, dataloader, SAEs, cache_dir, layer_pairs, NUM_FEATURES)