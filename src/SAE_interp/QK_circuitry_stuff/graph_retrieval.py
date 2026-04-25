import gc
import json
import os
import torch
from tqdm import tqdm
from transformers import ViTForImageClassification
from src.SAE.train_sae import TopKSparseAutoencoder, SparseAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_all_scores_chunked(
        V_in: torch.Tensor,  # Source features (L-1), shape: (F_in, D)
        V_out: torch.Tensor,  # Destination features (L), shape: (F_out, D)
        A_matrices: torch.Tensor,  # Attention cache, shape: (num_heads, F_out, F_in) -> A[h, j, i] = a_ji
        W_ov_heads: torch.Tensor,  # W_OV for targeted heads, shape: (num_heads, D, D)
        W_l: torch.Tensor,  # LayerNorm weight, shape: (D,)
        sigma_l: float,  # Standard deviation scalar
        P_j_given_i: torch.Tensor,  # NEW: The probability multiplier matrix
        chunk_size: int = 128  # Chunking over destination features to manage VRAM
) -> torch.Tensor:
    F_out, D = V_out.shape
    F_in = V_in.shape[0]
    num_heads = W_ov_heads.shape[0]

    # Final matrix to store Score_{l-1, i -> l, j}
    # Rows = j (destination), Cols = i (source)
    final_scores = torch.zeros((F_out, F_in), device=device, dtype=torch.float32)

    # Pre-calculate W_ov^(h) * v_i for all heads and source features
    # V_in is (F_in, D), W_ov_heads[h] is (D, D) -> Result: (num_heads, F_in, D)
    W_ov_V_in = torch.stack([torch.matmul(V_in, W_ov_heads[h].T) for h in range(num_heads)])

    # Process destination features in chunks
    for start_idx in tqdm(range(0, F_out, chunk_size), desc="Calculating Scores"):
        end_idx = min(start_idx + chunk_size, F_out)

        # v_j vectors for this chunk. Shape: (C, D) where C is chunk_size
        v_j_chunk = V_out[start_idx:end_idx]

        # a_ji attention weights for this chunk. Shape: (num_heads, C, F_in)
        A_chunk = A_matrices[:, start_idx:end_idx, :].to(device=device, dtype=torch.float32)

        # 1. Sum over heads: V = sum_{h} a_ji * (f_i * W_ov^(h) * v_i)
        # A_chunk is (num_heads, C, F_in)  -> 'h c i'
        # scaled_W_ov_V_in is (num_heads, F_in, D) -> 'h i d'
        # We want V of shape (C, F_in, D) -> 'c i d'
        V = torch.einsum('hci, hid -> cid', A_chunk, W_ov_V_in)

        # 2. Calculate Proj_{->1}(V)
        # Projection onto vector of ones is the mean across the D dimension
        V_mean = V.mean(dim=-1, keepdim=True)  # Shape: (C, F_in, 1)

        # 3. Calculate LayerNorm: LN_l(V) = b_l + W_l * (V - Proj_{->1}(V)) / sigma_l
        LN_V = W_l * (V - V_mean) / sigma_l  # Shape: (C, F_in, D)

        # 4. Final Score Projection: Proj_{v_j}[LN_l(V)]
        score_chunk = torch.einsum('cid, cd -> ci', LN_V, v_j_chunk)

        # --- NEW: Apply the Empirical Probability Gating ---
        P_chunk = P_j_given_i[start_idx:end_idx, :].to(device)
        score_chunk = score_chunk * P_chunk

        # Store in the final matrix
        final_scores[start_idx:end_idx, :] = score_chunk

    return final_scores


if __name__ == "__main__":
    # --- 1. Settings & Paths ---
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    SAE_dir = "C:\\Users\\ast12\\PycharmProjects\\CMPE492\\saved_models"
    SAE_stats_dir = f"C:\\Users\\ast12\\PycharmProjects\\CMPE492\\model_activation_stats"
    cache_dir = "C:\\Users\\ast12\\PycharmProjects\\CMPE492\\results\\attn_caches"
    model = ViTForImageClassification.from_pretrained(model_id).to(device)
    model.eval()

    for layer_idx in range(11, 0, -1):
        l_minus_1_node = f"lnb{layer_idx - 1}"
        l_node = f"lnb{layer_idx}"

        expansion_factor = 16
        l1_coefficient = "0.0001"  # Update this to match your exact saved filename string
        D = 768

        config = model.config
        head_dim = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads
        target_layer = model.vit.encoder.layer[layer_idx]

        # W_l and b_l
        W_l = target_layer.layernorm_before.weight.detach().to(device)
        b_l = target_layer.layernorm_before.bias.detach().to(device)

        W_value = target_layer.attention.attention.value.weight.detach().to(device)
        W_dense = target_layer.attention.output.dense.weight.detach().to(device)

        # --- 3. Load SAE Features (v_i and v_j) using regular SparseAutoencoder ---
        sae_l_minus_1_path = os.path.join(SAE_dir, f"sae_{l_minus_1_node}_ef{expansion_factor}_l1{l1_coefficient}.pt")
        sae_l_path = os.path.join(SAE_dir, f"sae_{l_node}_ef{expansion_factor}_l1{l1_coefficient}.pt")

        sae_l_minus_1 = SparseAutoencoder(input_dim=D, expansion_factor=expansion_factor).to(device)
        sae_l_minus_1.load_state_dict(torch.load(sae_l_minus_1_path)["model_state_dict"])
        with open(os.path.join(SAE_stats_dir, f"sae_{l_minus_1_node}_ef{expansion_factor}_l1{l1_coefficient}.json"), "r") as f:
            sae_l_minus_1_stats = json.load(f)["feature_means_nz"]

        sae_l = SparseAutoencoder(input_dim=D, expansion_factor=expansion_factor).to(device)
        sae_l.load_state_dict(torch.load(sae_l_path)["model_state_dict"])

        V_in = sae_l_minus_1.W_dec.detach().to(device)  # v_i
        V_out = sae_l.encoder.weight.detach().to(device)  # Transposed to match (F_out, D)
        F_in = V_in.shape[0]

        # --- 4. Load Attention Cache (a_ji) ---
        head_gorups = ["0-1-2-3-4-5", "6-7-8-9-10-11"]
        A_matrices = []
        target_heads = []

        # Trackers for the counts
        pair_counts = None
        src_counts = None

        for head_group in head_gorups:
            attn_cache_path = os.path.join(cache_dir,
                                           f"full_empirical_attn_{l_minus_1_node}-{l_node}_heads_{head_group}.pt")
            attn_data = torch.load(attn_cache_path, map_location="cpu")

            target_heads.extend(attn_data["target_heads"])
            A_matrices.append(attn_data["avg_attention"].float())
            sigma_l_val = attn_data["avg_layer_std"]

            # Load the counts only once from the first file
            if pair_counts is None:
                pair_counts = attn_data["pair_counts"].float()
                # Note: using the correct key saved from full_empatt_save.py
                src_counts = attn_data["feature_counts_src"].float()

        A_matrices = torch.cat(A_matrices, dim=0)

        # --- NEW: Calculate the P(j | i) matrix ---
        # pair_counts shape: (F_out, F_in)
        # src_counts shape: (F_in,) -> unsqueeze makes it (1, F_in) for broadcasting
        # We clamp the denominator to 1.0 to prevent division by zero
        P_j_given_i = (pair_counts / torch.clamp(src_counts.unsqueeze(0), min=1.0)).clamp(max=1.0)

        W_ov_target_heads = []
        for h in target_heads:
            W_V_h = W_value[h * head_dim: (h + 1) * head_dim, :]
            W_O_h = W_dense[:, h * head_dim: (h + 1) * head_dim]
            W_ov_target_heads.append(W_O_h @ W_V_h)

        W_ov_target_heads = torch.stack(W_ov_target_heads)

        print(f"Starting calculation for Layer {layer_idx - 1} -> Layer {layer_idx}")
        full_score_matrix = calculate_all_scores_chunked(
            V_in=V_in,
            V_out=V_out,
            A_matrices=A_matrices,
            W_ov_heads=W_ov_target_heads,
            W_l=W_l,
            sigma_l=sigma_l_val,
            P_j_given_i=P_j_given_i,  # Pass our new matrix here!
            chunk_size=32
        )

        print(f"Finished! Score Matrix Shape: {full_score_matrix.shape}")

        save_out_path = os.path.join(cache_dir,f"interaction_scores_{l_minus_1_node}_to_{l_node}.pt")
        torch.save(full_score_matrix.cpu(), save_out_path)
        print(f"Saved scores to {save_out_path}")

        torch.cuda.empty_cache()
        gc.collect()