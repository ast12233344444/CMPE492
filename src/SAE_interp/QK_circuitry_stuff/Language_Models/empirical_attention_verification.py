import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)


def sample_feature_pairs_llm(interaction_matrix, n_split, n_feature_per_split):
    """
    Samples feature pairs (q_feat, k_feat) from different strength percentiles.
    Optimized for memory: avoids creating massive N x 2 index arrays.
    """
    assert n_split >= 2

    flat_matrix = interaction_matrix.flatten().detach().cpu()

    sorted_values, sorted_1d_indices = torch.sort(flat_matrix)

    highest_bound = sorted_values[-n_feature_per_split].item()
    lowest_bound = sorted_values[n_feature_per_split].item()
    bounds = np.linspace(lowest_bound, highest_bound, n_split - 1)

    bound_samples = []
    num_features = interaction_matrix.shape[1]
    rng = np.random.default_rng()

    lowest_slice_1d = sorted_1d_indices[:n_feature_per_split]
    row_indices = (lowest_slice_1d // num_features).numpy()
    col_indices = (lowest_slice_1d % num_features).numpy()
    bound_samples.append(np.column_stack((row_indices, col_indices)))

    for i in tqdm(range(1, len(bounds)), desc="Sampling exact bins"):
        bound_h = bounds[i]
        bound_h_i = torch.searchsorted(sorted_values, bound_h).item()

        bound_l = bounds[i - 1]
        bound_l_i = torch.searchsorted(sorted_values, bound_l).item()

        slice_idx_1d = sorted_1d_indices[bound_l_i: bound_h_i].numpy()

        sample_size = min(n_feature_per_split, len(slice_idx_1d))
        if sample_size > 0:
            sampled_1d = rng.choice(slice_idx_1d, size=sample_size, replace=False)

            row_indices = sampled_1d // num_features
            col_indices = sampled_1d % num_features
            bound_samples.append(np.column_stack((row_indices, col_indices)))
        else:
            bound_samples.append(np.empty((0, 2), dtype=int))

    highest_slice_1d = sorted_1d_indices[-n_feature_per_split:]
    row_indices = (highest_slice_1d // num_features).numpy()
    col_indices = (highest_slice_1d % num_features).numpy()
    bound_samples.append(np.column_stack((row_indices, col_indices)))

    del flat_matrix
    del sorted_values
    del sorted_1d_indices

    return bounds, bound_samples


def get_empirical_attention_llm(LLM, SAE, layer_idx, head_idx, dataloader, feature_pairs, interaction_matrix, L0_norm, attention_df, max_dist=None):
    """
    Calculates the average empirical causal attention score from source tokens (k) to
    destination tokens (q) whenever specific SAE feature pairs are co-active within a specific distance.
    """
    if not hasattr(LLM, 'trace'):
        from nnsight import LanguageModel
        LLM = LanguageModel(LLM)

    pairs = [tuple(pair) for pair in feature_pairs]
    attention_sums = {pair: 0.0 for pair in pairs}
    attention_counts = {pair: 0 for pair in pairs}

    with torch.no_grad():
        for batch in tqdm(dataloader,
                          desc=f"Getting Empirical Attention (L{layer_idx}, H{head_idx}, Max Dist: {max_dist})"):
            input_ids = batch["input_ids"].to(device)
            batch_size, seq_len_input = input_ids.shape

            with LLM.trace(input_ids) as tracer:
                hidden_states = LLM.model.layers[layer_idx].input[0].save()
                attention_probs = LLM.model.layers[layer_idx].self_attn.output[1].save()

            feats_present = (SAE.encode(hidden_states.value) > 0)
            feats_present = feats_present.view(batch_size, seq_len_input, -1)
            attns = attention_probs.value[:, head_idx, :, :]
            seq_len = attns.shape[-1]

            del hidden_states
            del attention_probs

            # --- MODIFIED MASK LOGIC ---
            causal_mask = torch.tril(
                torch.ones((seq_len, seq_len), device=attns.device, dtype=torch.bool)
            )

            if max_dist is not None:
                window_mask = torch.triu(
                    torch.ones((seq_len, seq_len), device=attns.device, dtype=torch.bool),
                    diagonal=-max_dist
                )
                causal_mask = causal_mask & window_mask
            # ---------------------------

            for f_q, f_k in pairs:
                q_present = feats_present[:, :, f_q]
                k_present = feats_present[:, :, f_k]

                pair_mask = q_present.unsqueeze(2) & k_present.unsqueeze(1)
                valid_pair_mask = pair_mask & causal_mask.unsqueeze(0)

                attention_sums[(f_q, f_k)] += (attns * valid_pair_mask).sum().item()
                attention_counts[(f_q, f_k)] += valid_pair_mask.sum().item()

    empirical_attentions = {}
    att_mean = interaction_matrix.mean()
    att_std = interaction_matrix.std()
    for pair in pairs:
        pair_dest, pair_src = pair
        th_score = (interaction_matrix[pair] - att_mean) / (att_std * L0_norm)
        data = {
            "feature_source": pair_src,
            "feature_dest": pair_dest,
            "layer": layer_idx,
            "head": head_idx,
            "theoretical_significance_score": th_score.item(),
            "mean_empirical_attention": None,
            "no_samples_gathered": attention_counts[pair],
        }
        if attention_counts[pair] > 0:
            empirical_attentions[pair] = attention_sums[pair] / attention_counts[pair]
        else:
            empirical_attentions[pair] = 0.0
        data["mean_empirical_attention"] = empirical_attentions[pair]
        attention_df.loc[len(attention_df)] = data

    return empirical_attentions


def get_cached_streaming_batches(tokenizer, batch_size=4, context_length=2048, num_batches=50):
    """
    Streams from FineWeb-Edu, tokenizes, and caches a fixed number of batches
    in memory so they can be reused across multiple layer/head evaluations.
    """
    print("Loading FineWeb-Edu dataset (streaming) and caching batches...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    dataset_iter = iter(dataset)

    batches = []
    for _ in tqdm(range(num_batches), desc="Caching text batches"):
        text_batch = []
        while len(text_batch) < batch_size:
            try:
                text_batch.append(next(dataset_iter)["text"])
            except StopIteration:
                break

        if not text_batch:
            break

        inputs = tokenizer(
            text_batch,
            return_tensors="pt",
            max_length=context_length,
            truncation=True,
            padding="max_length"
        )
        batches.append(inputs)

    return batches