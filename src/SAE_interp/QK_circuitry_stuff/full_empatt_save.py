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


def get_stratified_subset(dataset, num_samples_per_class=10):
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


def cache_full_empirical_attention(model, layer_name_dest, layer_name_src, layer_i, target_heads, num_features,
                                   SAE_dest, SAE_src, dataloader, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    heads_str = "-".join([str(h) for h in target_heads])
    save_path = os.path.join(out_dir, f"full_empirical_attn_{layer_name_src}-{layer_name_dest}_heads_{heads_str}.pt")

    num_target_heads = len(target_heads)
    total_sums = torch.zeros((num_target_heads, num_features, num_features), dtype=torch.float32, device=device)
    total_counts = torch.zeros((num_features, num_features), dtype=torch.float32, device=device)

    # Trackers for average standard deviation across all tokens
    total_std_sum = 0.0
    total_std_count = 0

    if not hasattr(model, 'trace'):
        from nnsight import NNsight
        model = NNsight(model)

    model.config.output_attentions = True
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    with torch.no_grad():
        for batch in tqdm(dataloader,
                          desc=f"Caching Attention ({layer_name_src} -> {layer_name_dest}, Heads {heads_str})"):
            x_batch, _ = batch
            x_batch = x_batch.to(model.device)
            current_batch_size = x_batch.size(0)

            with model.trace() as tracer:
                with tracer.invoke(x_batch) as invoker:
                    # 1. Extract Activations from BOTH layers
                    act_dest = TracingAlgorithms._get_activations(model, layer_name_dest, head_dim)
                    act_src = TracingAlgorithms._get_activations(model, layer_name_src, head_dim)

                    ln_input = model.vit.encoder.layer[layer_i].layernorm_before.input[0]
                    act_dest_std = ln_input.std(dim=-1, unbiased=False).save()

                    encoded_dest, _ = SAE_dest(act_dest)
                    encoded_src, _ = SAE_src(act_src)

                    encoded_dest = encoded_dest.save()
                    encoded_src = encoded_src.save()

                    attention_probs = model.vit.encoder.layer[layer_i].attention.attention.output[1].save()

            # Accumulate Standard Deviation data
            std_val = act_dest_std.value
            total_std_sum += std_val.sum().item()
            total_std_count += std_val.numel()  # Number of tokens total in this batch

            feats_dest = encoded_dest.value
            feats_src = encoded_src.value
            attns = attention_probs.value

            # Binarize feature presence
            F_dest_present = (feats_dest > 0).float()
            F_src_present = (feats_src > 0).float()

            for b in range(current_batch_size):
                F_d = F_dest_present[b]  # (Seq_Len, D_dest)
                F_s = F_src_present[b]  # (Seq_Len, D_src)

                F_d_sum = F_d.sum(dim=0)  # (D_dest,)
                F_s_sum = F_s.sum(dim=0)  # (D_src,)

                if F_d_sum.sum() == 0 or F_s_sum.sum() == 0:
                    continue

                # Cross-layer co-occurrence count
                total_counts.addr_(F_d_sum, F_s_sum)

                for i, h in enumerate(target_heads):
                    A_bh = attns[b, h]  # (Seq_Len_dest, Seq_Len_src)

                    # temp: (Seq_Len_dest, D_src)
                    temp = torch.matmul(A_bh, F_s)

                    # Add to total sums: (D_dest, D_src)
                    total_sums[i].addmm_(F_d.t(), temp)

            del feats_dest, feats_src, attns, F_dest_present, F_src_present, encoded_dest, encoded_src, attention_probs
            gc.collect()
            torch.cuda.empty_cache()
        print("Moving accumulators to CPU and calculating final averages...")

        total_sums = total_sums.cpu()
        total_counts = total_counts.cpu()
        torch.cuda.empty_cache()
        safe_counts = total_counts.clamp(min=1.0)
        total_sums.div_(safe_counts.unsqueeze(0))
        del safe_counts  # Free the clamped tensor
        zero_mask = (total_counts == 0).unsqueeze(0).expand(num_target_heads, -1, -1)
        total_sums[zero_mask] = 0.0
        del zero_mask  # Free the mask
        avg_attention = total_sums.to(torch.float16)
        del total_sums  # Free the float32 tensor

        # Calculate final average standard deviation
        avg_std = total_std_sum / total_std_count if total_std_count > 0 else 1.0

        print(f"Saving to disk at {save_path}... avg std: {avg_std}")
        torch.save({
            "target_heads": target_heads,
            "avg_attention": avg_attention,
            "pair_counts": total_counts.to(torch.int32),
            "avg_layer_std": float(avg_std)
        }, save_path)

        print("Done!")
        return avg_attention


if __name__ == "__main__":

    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager").to(device)
    processor = ViTImageProcessor.from_pretrained(model_id)
    model.eval()
    batch_size = 16

    dataset_full = datasets.CIFAR10(root='..\\data', train=True, download=True)
    subset = get_stratified_subset(dataset_full, num_samples_per_class=1000)
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
    HEADSETS_TO_CACHE = [[i, i+1, i+2, i+3, i+4, i+5] for i in range(0, 12, 6)]

    cache_dir = "C:\\Users\\ast12\\PycharmProjects\\CMPE492\\results\\attn_caches"

    for layer in tqdm(range(7, 0, -1), "Caching layers"):
        for HEADS_TO_CACHE in HEADSETS_TO_CACHE:
            layer_name_dest = f"lnb{layer}"
            layer_name_src = f"lnb{layer - 1}"

            avg_attention_cache = cache_full_empirical_attention(
                model=model,
                layer_name_dest=layer_name_dest,
                layer_name_src=layer_name_src,
                layer_i=layer,
                target_heads=HEADS_TO_CACHE,
                num_features=NUM_FEATURES,
                SAE_dest=SAEs[layer_name_dest],
                SAE_src=SAEs[layer_name_src],
                dataloader=dataloader,
                out_dir=cache_dir
            )