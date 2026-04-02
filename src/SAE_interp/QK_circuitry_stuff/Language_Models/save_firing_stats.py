import gc

import torch
from datasets import load_dataset
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

def measure_scaled_feature_intensities(
        model_id: str,
        sae_id_template: str,
        sae_release: str,
        layer_idxs: list[int],
        max_tokens: int = 100_000_000,
        batch_size: int = 4,
        context_length: int = 2048
):

    print(f"Loading Tokenizer and Model ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    model.eval()

    # Extract the RMSNorm epsilon directly from the model config
    rms_eps = model.config.rms_norm_eps

    saes = {}
    for layer_idx in layer_idxs:
        sae_id = sae_id_template.replace("[L]", str(layer_idx))
        print(f"Loading SAE ({sae_id})...")
        sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
            release= sae_release,
            sae_id=sae_id,
            device=device_str,
        )
        sae.eval()
        saes[layer_idx] = sae

    print("Loading FineWeb-Edu dataset (streaming)...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    dataset_iter = iter(dataset)

    num_features = sae.cfg.d_sae

    feature_firing_counts = {}
    feature_activation_sums = {}
    for layer_idx in layer_idxs:
        feature_firing_counts[layer_idx] = torch.zeros(num_features, dtype=torch.long, device=device)
        feature_activation_sums[layer_idx] = torch.zeros(num_features, dtype=torch.float64, device=device)
    total_valid_tokens = 0

    print(f"Beginning evaluation... Target: {max_tokens:,} tokens.")
    pbar = tqdm(total=max_tokens, desc="Processing Tokens")

    text_batch = []

    with torch.no_grad():
        while total_valid_tokens < max_tokens:
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
            ).to(device)

            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            for layer_idx in layer_idxs:
                hidden_states = outputs.hidden_states[layer_idx + 1]

                variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
                rms_scale = torch.rsqrt(variance + rms_eps).to(hidden_states.dtype)

                feature_acts = saes[layer_idx].encode(hidden_states)

                scaled_feature_acts = feature_acts * rms_scale

                valid_tokens_mask = inputs.attention_mask.unsqueeze(-1).bool()

                fired_mask = (feature_acts > 0) & valid_tokens_mask
                feature_firing_counts[layer_idx] += fired_mask.sum(dim=(0, 1))

                valid_scaled_acts = scaled_feature_acts * valid_tokens_mask.to(scaled_feature_acts.dtype)
                feature_activation_sums[layer_idx] += valid_scaled_acts.sum(dim=(0, 1)).to(torch.float64)

                del hidden_states, rms_scale, feature_acts, scaled_feature_acts, fired_mask, valid_scaled_acts, valid_tokens_mask, variance

            batch_valid_tokens = inputs.attention_mask.sum().item()
            total_valid_tokens += batch_valid_tokens
            pbar.update(batch_valid_tokens)

            text_batch = []
            del inputs, outputs

    pbar.close()

    print("\nCalculating final scaled intensities...")

    average_nonzero_scaled_intensities = {}
    for layer_idx in layer_idxs:
        average_nonzero_scaled_intensities[layer_idx] = torch.zeros(num_features, dtype=torch.float32, device=device)
        alive_mask = feature_firing_counts[layer_idx] > 0

        average_nonzero_scaled_intensities[layer_idx][alive_mask] = (
                feature_activation_sums[layer_idx][alive_mask] / feature_firing_counts[layer_idx][alive_mask].to(torch.float64)
        ).to(torch.float32)

        print("\n--- Results ---")
        print(f"Total Valid Tokens Processed: {total_valid_tokens:,}")

        dead_features = (~alive_mask).sum().item()
        print(f"Dead Features: {dead_features} ({dead_features / num_features * 100:.1f}%)")

        live_intensities = average_nonzero_scaled_intensities[layer_idx][alive_mask]
        if len(live_intensities) > 0:
            print(f"Mean scaled intensity of alive features: {live_intensities.mean().item():.3f}")
            print(f"Max scaled intensity of alive features: {live_intensities.max().item():.3f}")

    return average_nonzero_scaled_intensities

if __name__ == "__main__":
    layer_to_investigate = 17
    head_to_investigate = 0
    assert layer_to_investigate > 0

    target_release = "gemma-scope-2-270m-pt-res-all"
    sae_id = f"layer_{layer_to_investigate-1}_width_16k_l0_big"
    model_id = "google/gemma-3-270m"
    L0_norm = 100

    print(f"Loading SAE: {sae_id}...")

    for layer_idx in range(5, 6):
        layers_to_get_firing = [layer_idx]
        densities = measure_scaled_feature_intensities(
            model_id="google/gemma-3-270m",
            sae_release="gemma-scope-2-270m-pt-res-all",
            sae_id_template=f"layer_[L]_width_16k_l0_big",
            layer_idxs=layers_to_get_firing,
            max_tokens=10_000_000,
            batch_size=4,
            context_length=2048
        )

        for layer_idx, density in densities.items():
            torch.save(density, f"SAE_firing_stats/feature_densities_SAE_l{layer_idx}_100M.pt")
            print(f"\nSaved densities to 'SAE_firing_stats/feature_densities_SAE_l{layer_idx}_100M.pt'")
        print("Cleaning up resources...")
        del densities
        gc.collect()
        torch.cuda.empty_cache()
        print("Done!")
