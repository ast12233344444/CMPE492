import json
import os

import numpy as np
import torch
import gc

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

from src.SAE.train_sae import SparseAutoencoder
from src.TracingAlgorithms import TracingAlgorithms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels


def get_stratified_subset(dataset, num_samples_per_class=500):
    """
    Creates a perfectly balanced subset of CIFAR-10.
    num_samples_per_class=500 yields 5,000 images total (10% of CIFAR-10).
    """
    targets = np.array(dataset.targets)
    indices = []

    # CIFAR-10 has 10 classes (0 through 9)
    for c in range(10):
        c_idx = np.where(targets == c)[0]
        # Randomly choose the specified number of samples without replacement
        chosen = np.random.choice(c_idx, num_samples_per_class, replace=False)
        indices.extend(chosen)

    return Subset(dataset, indices)


def cache_full_empirical_attention(model, layer_name, layer_i, target_heads, num_features, SAE, dataloader, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Create a string representation of the heads (e.g., "0-1-2-3") to prevent overwriting
    heads_str = "-".join([str(h) for h in target_heads])
    save_path = os.path.join(out_dir, f"full_empirical_attn_{layer_name}_heads_{heads_str}.pt")

    if os.path.exists(save_path):
        print(f"Cache already exists at {save_path}. Skipping.")
        return torch.load(save_path)

    num_target_heads = len(target_heads)

    # By limiting to target_heads, we drastically cut down RAM usage
    total_sums = torch.zeros((num_target_heads, num_features, num_features), dtype=torch.float64, device='cpu')
    total_counts = torch.zeros((num_features, num_features), dtype=torch.float64, device='cpu')

    # Wrap in NNsight if it isn't already
    if not hasattr(model, 'trace'):
        from nnsight import NNsight
        model = NNsight(model)

    model.config.output_attentions = True
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Caching Attention ({layer_name}, Heads {heads_str})"):
            x_batch, _ = batch
            x_batch = x_batch.to(model.device)
            current_batch_size = x_batch.size(0)

            with model.trace() as tracer:
                with tracer.invoke(x_batch) as invoker:
                    activations = TracingAlgorithms._get_activations(model, layer_name, head_dim)

                    encoded_features, _ = SAE(activations)
                    encoded_features = encoded_features.save()

                    # Shape: (Batch, Total_Heads, Seq_Len, Seq_Len)
                    attention_probs = model.vit.encoder.layer[layer_i].attention.attention.output[1].save()

            feats = encoded_features.value
            attns = attention_probs.value

            # Binarize feature presence: (Batch, Seq_Len, Features)
            feats_present = (feats > 0).float()

            # Process sequentially by batch item
            for b in range(current_batch_size):
                F_b = feats_present[b]  # (Seq_Len, D)

                # 1. Find features that fired AT LEAST ONCE in this sequence
                F_b_sum = F_b.sum(dim=0)  # (D,)
                active_idx = F_b_sum.nonzero(as_tuple=True)[0]

                # If no features fired (extremely rare, but safe to check), skip
                if active_idx.numel() == 0:
                    continue

                # 2. Slice out only the active features. Shape becomes (Seq_Len, K)
                F_b_active = F_b[:, active_idx]
                F_b_sum_active = F_b_sum[active_idx]

                # Prepare CPU indexing grids for the accumulators
                row_idx = active_idx.unsqueeze(1).cpu()
                col_idx = active_idx.unsqueeze(0).cpu()

                # 3. Sparse Count Accumulation
                counts_update = torch.outer(F_b_sum_active, F_b_sum_active).cpu().double()
                total_counts[row_idx, col_idx] += counts_update

                # 4. Sparse Attention Accumulation ONLY for target heads
                for i, h in enumerate(target_heads):
                    A_bh = attns[b, h]  # (Seq_Len, Seq_Len)

                    # Math equivalent: (K, Seq) @ (Seq, Seq) @ (Seq, K) => (K, K)
                    sum_bh_active = (F_b_active.t() @ A_bh @ F_b_active).cpu().double()

                    # Map the (K, K) sums directly back to the correct global indices at our mapped head index `i`
                    total_sums[i, row_idx, col_idx] += sum_bh_active

            # Memory management
            del feats, attns, feats_present, encoded_features, attention_probs
            gc.collect()
            torch.cuda.empty_cache()

    # Calculate final averages
    print("Calculating final averages and casting back to float16...")

    # Avoid division by zero
    safe_counts = total_counts.clamp(min=1.0)
    avg_attention = total_sums / safe_counts.unsqueeze(0)

    # Zero out positions where the pair count was actually 0
    zero_mask = (total_counts == 0).unsqueeze(0).expand(num_target_heads, -1, -1)
    avg_attention[zero_mask] = 0.0

    # Downcast back to float16 for storage
    avg_attention = avg_attention.to(torch.float16)

    print(f"Saving to disk at {save_path}...")
    torch.save({
        "target_heads": target_heads,  # Save the head mapping so you know what is in this file
        "avg_attention": avg_attention,
        "pair_counts": total_counts.to(torch.int32)
    }, save_path)

    print("Done!")
    return avg_attention


if __name__ == "__main__":

    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager").to(device)
    processor = ViTImageProcessor.from_pretrained(model_id)
    model.eval()
    batch_size = 64

    dataset_full = datasets.CIFAR10(root='../data', train=True, download=True)
    subset = get_stratified_subset(dataset_full, num_samples_per_class=1000)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    SAE_dir = "/home/ahmet/PycharmProjects/CMPE492/saved_models"
    SAE_act_stats_dir = f"/home/ahmet/PycharmProjects/CMPE492/model_activation_stats"
    results_dir = f"/home/ahmet/PycharmProjects/CMPE492/results/QK_circuit_analysis"
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
    HEADSETS_TO_CACHE = [[0, 1, 2, 3], [4,5,6,7], [8,9,10,11]]

    cache_dir = "/home/ahmet/PycharmProjects/CMPE492/results/QK_circuit_analysis/caches"

    for layer in tqdm(range(11, -1, -1), "Caching layers"):
        for HEADS_TO_CACHE in HEADSETS_TO_CACHE:
            layer_name = f"lnb{layer}"

            # Run the caching script
            avg_attention_cache = cache_full_empirical_attention(
                model=model,
                layer_name=layer_name,
                layer_i=layer,
                target_heads=HEADS_TO_CACHE,  # Pass the configurable heads here
                num_features=NUM_FEATURES,
                SAE=SAEs[layer_name],
                dataloader=dataloader,
                out_dir=cache_dir
            )