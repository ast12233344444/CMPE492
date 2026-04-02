import gc
import math
import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sae_lens import SAE

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.SAE_interp.QK_circuitry_stuff.Language_Models.empirical_attention_verification import \
    get_cached_streaming_batches, sample_feature_pairs_llm, get_empirical_attention_llm

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

class rotary_emb():
    @staticmethod
    def rotate_half(x):
        """Matches Hugging Face's Half-Split dimension grouping."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def get_cos_sin(dim, offset, base_theta, device, dtype):
        """Generates frequencies and concatenates them the Gemma 3 way."""
        inv_freq = 1.0 / (base_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freqs = float(offset) * inv_freq

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype), sin.to(dtype)

    @staticmethod
    def rotate(u, offset, base_theta):
        dim = u.shape[-1]
        cos, sin = rotary_emb.get_cos_sin(dim, offset, base_theta, u.device, u.dtype)
        return (u * cos) + (rotary_emb.rotate_half(u) * sin)


def get_feature_feature_interactions(LLM, SAE, sae_firings, layer_idx: int, head_idx: int, max_lag=128):
    """
    Calculates the feature-feature interaction matrix for each attention head
    based on the pre-RMSNorm SAE features and the model's QK circuitry.
    """
    config = LLM.model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_groups = num_heads // num_kv_heads

    V = SAE.W_dec.detach().to(device)
    interaction_h = torch.zeros(V.shape[0], V.shape[0]).to(device)

    layer = LLM.model.layers[layer_idx]

    w_raw = layer.input_layernorm.weight.detach().to(device)
    W_RMS = 1.0 + w_raw

    V_normed = V * W_RMS * sae_firings.unsqueeze(1)

    q_weight = layer.self_attn.q_proj.weight.detach().to(device)
    k_weight = layer.self_attn.k_proj.weight.detach().to(device)

    W_Q_h = q_weight[head_idx * head_dim: (head_idx + 1) * head_dim, :]

    # Map Query head to its corresponding Key head (GQA)
    h_kv = head_idx // num_groups
    W_K_h = k_weight[h_kv * head_dim: (h_kv + 1) * head_dim, :]

    # Project the normed features into Q and K space
    # Shape: [d_sae, head_dim]
    Q_feat = V_normed @ W_Q_h.T
    K_feat = V_normed @ W_K_h.T

    i = 0
    for lag in tqdm(range(0, max_lag, math.ceil(max_lag/100)), "calculating interaction matrix..."):
        i += 1
        if layer.self_attn.is_sliding:
            base_theta = config.rope_local_base_freq
        else:
            base_theta = config.rope_theta

        Q_feat_rot = rotary_emb.rotate(Q_feat, offset=lag, base_theta=base_theta)

        temp = torch.matmul(Q_feat_rot, K_feat.T)
        temp.div_(head_dim ** 0.5)  # The underscore means "in-place division"
        interaction_h.add_(temp)  # In-place addition
        del temp  # Free immediately
        torch.cuda.empty_cache()
        gc.collect()

    interaction_h /= i
    return interaction_h

def get_feature_attractions_by_lag(LLM, feat_vec_pairs, layer_idx, head_idx, maxlag = 10000, out_dir = f""):
    config = LLM.model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_groups = num_heads // num_kv_heads

    feat_interaction_forces = {pair: [] for pair in feat_vec_pairs}
    for lag in range(0, maxlag, math.ceil(max_lag/100)):
        for pair in feat_vec_pairs:
            Q, K = feat_vec_pairs[pair]

            layer = LLM.model.layers[layer_idx]

            w_raw = layer.input_layernorm.weight.detach().to(device)
            W_RMS = 1.0 + w_raw

            Q_normed = Q * W_RMS
            K_normed = K * W_RMS

            Q_normed_Linf = Q_normed.abs().max()
            Q_normed_L1 = Q_normed.abs().sum()

            K_normed_Linf = K_normed.abs().max()
            K_normed_L1 = K_normed.abs().sum()

            q_weight = layer.self_attn.q_proj.weight.detach().to(device)
            k_weight = layer.self_attn.k_proj.weight.detach().to(device)

            W_Q_h = q_weight[head_idx * head_dim: (head_idx + 1) * head_dim, :]

            # Map Query head to its corresponding Key head (GQA)
            h_kv = head_idx // num_groups
            W_K_h = k_weight[h_kv * head_dim: (h_kv + 1) * head_dim, :]

            Q_feat = Q_normed @ W_Q_h.T
            K_feat = K_normed @ W_K_h.T

            if lag > 0:
                if layer.self_attn.is_sliding:
                    base_theta = config.rope_local_base_freq
                else:
                    base_theta = config.rope_theta

                Q_feat = rotary_emb.rotate(Q_feat, lag, base_theta)

            interaction_h = (Q_feat @ K_feat.T) / (head_dim ** 0.5)
            feat_interaction_forces[pair].append(interaction_h.item())

    plt.figure(figsize=(16, 9))
    lags = range(0, maxlag, math.ceil(max_lag/100))
    for pair in feat_vec_pairs:
        plt.plot(lags, feat_interaction_forces[pair],
                 label=f"{pair}, rats: {Q_normed_Linf/Q_normed_L1:4}, {K_normed_Linf/K_normed_L1:4}")
    plt.title(f"layer {layer_idx} head {head_idx} top feature interactions by lag")
    plt.legend()
    #plt.savefig(out_dir)

    plt.show()

def make_hist(tensor):
    L0_norm = 100
    flat_matrix = tensor.flatten().detach().cpu().numpy()
    matmean = np.mean(flat_matrix)
    matstd = np.std(flat_matrix)
    lbound = matmean - 2 * L0_norm * matstd
    hbound = matmean + 2 * L0_norm * matstd

    plt.figure(figsize=(10, 6))
    plt.hist(flat_matrix[(flat_matrix > lbound) & (flat_matrix < hbound)], bins=150, log=True, color='skyblue',
             alpha=0.7)
    plt.hist(flat_matrix[(flat_matrix < lbound) | (flat_matrix > hbound)], bins=150, log=True, color='red', alpha=0.7)

    plt.title(f"Distribution of QK feature attractions (Layer {layer_to_investigate}, Head {head_to_investigate})",
              fontsize=14)
    plt.xlabel("Interaction Strength", fontsize=12)
    plt.ylabel("Frequency (Log Scale)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    layer_to_investigate = 3
    head_to_investigate = 2
    assert layer_to_investigate > 0
    empirical_lag = 128
    empirical_df_path = "/home/ahmet/PycharmProjects/CMPE492/results/Gemma/empirical_attention.csv"

    if not os.path.exists(empirical_df_path):
        empirical_attention_df = pd.DataFrame({
            "feature_source": [],
            "feature_dest": [],
            "layer": [],
            "head": [],
            "theoretical_significance_score": [],
            "mean_empirical_attention":[],
            "no_samples_gathered": []
        })
    else:
        empirical_attention_df = pd.read_csv(empirical_df_path)

    model_id = "google/gemma-3-270m"

    # 1. Load Tokenizer
    print(f"Loading Tokenizer ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = get_cached_streaming_batches(
        tokenizer,
        batch_size=1,
        context_length=1024,
        num_batches=2**9
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        output_attentions=True,
        attn_implementation="eager"
    )
    model.eval()
    print(f"model softcapping : {model.config.attn_logit_softcapping}")

    from nnsight import LanguageModel
    model = LanguageModel(model, tokenizer=tokenizer)
    L0_norm = 100

    for layer_to_investigate in range(17, -1, -1):
        target_release = "gemma-scope-2-270m-pt-res-all"
        sae_id = f"layer_{layer_to_investigate - 1}_width_16k_l0_big"
        sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
            release=target_release,
            sae_id=sae_id,
            device=device_str,
        )
        sae.eval()

        sae_firings = torch.load(f"/home/ahmet/PycharmProjects/CMPE492/src/SAE_interp/QK_circuitry_stuff"
                                 f"/Language_Models/SAE_firing_stats/feature_densities_SAE_l{layer_to_investigate - 1}_100M.pt")
        #sae_firings = torch.ones(2 ** 14, device=device)
        for head_to_investigate in range(4):
            out_folder = f"/home/ahmet/PycharmProjects/CMPE492/results/Gemma/RoPE_behaviour/l{layer_to_investigate}_h{head_to_investigate}/"
            os.makedirs(out_folder, exist_ok=True)

            tensor = get_feature_feature_interactions(model, sae, sae_firings, layer_to_investigate, head_to_investigate, max_lag=empirical_lag)
            make_hist(tensor)

            print("getting bound samples")
            bounds, bound_samples = sample_feature_pairs_llm(tensor, n_split=10, n_feature_per_split=100)
            bound_samples_flat = np.concatenate(bound_samples)

            print("getting empirical attention")
            empirical_attention_data = get_empirical_attention_llm(
                model, sae, layer_to_investigate, head_to_investigate, dataloader, bound_samples_flat,
                tensor, L0_norm, empirical_attention_df, max_dist=empirical_lag)
            empirical_attention_df.to_csv(empirical_df_path)

            bound_groups = [[] for _ in range(len(bound_samples))]
            for i in range(len(bound_samples)):
                for feature_pair in empirical_attention_data:
                    f_q, f_k = feature_pair
                    strength = tensor[f_q, f_k].item()

                    if i > 0 and strength < bounds[i - 1]: continue
                    if i < len(bound_samples) - 1 and strength > bounds[i]: continue

                    bound_groups[i].append(empirical_attention_data[feature_pair])

            valid_groups = [group for group in bound_groups if len(group) > 0]
            if valid_groups:
                plt.figure(figsize=(10, 6))
                plt.boxplot(valid_groups, showfliers=False, patch_artist=True)
                plt.title(
                    f"LLM Empirical Attention by QK Strength Bin (L{layer_to_investigate}, H{head_to_investigate})")
                plt.xlabel("QK Strength Bins (Lowest to Highest)")
                plt.ylabel("Average Causal Attention Probability")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                #plt.savefig(f"{out_folder}/empirical_attention.png")
                plt.show()

            flat_matrix = tensor.flatten().detach().cpu().numpy()
            matmean = np.mean(flat_matrix)
            matstd = np.std(flat_matrix)
            hbound = matmean + 2 * L0_norm * matstd
            locations = torch.nonzero(tensor > hbound).tolist()
            if len(locations) > 10:
                locations = random.sample(locations, 10)
            if len(locations) == 0:
                continue
            print(locations)
            feat_vec_pairs = {}
            for location in locations:
                feat_vec_pairs[f"{location}"] = (sae.W_dec[location[0]] * sae_firings[location[0]], sae.W_dec[location[1]] * sae_firings[location[1]])
            for max_lag in [128, 1000, 10000]:
                get_feature_attractions_by_lag(model, feat_vec_pairs, layer_to_investigate, head_to_investigate, maxlag = max_lag,
                    out_dir=f"{out_folder}/rope_curve_{max_lag}.png")

            del tensor
            torch.cuda.empty_cache()
            gc.collect()
