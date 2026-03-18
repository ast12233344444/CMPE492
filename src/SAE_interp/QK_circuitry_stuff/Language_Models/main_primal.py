import torch
from nnsight import LanguageModel
from sae_lens import SAE, HookedSAETransformer

from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

directory = get_pretrained_saes_directory()

print("--- Available Gemma Scope 2 270M Releases ---")
for release_name in directory.keys():
    if "gemma-scope-2-270m" in release_name:
        print(release_name)

target_release = "gemma-scope-2-270m-pt-res-all"
print(f"\n--- SAE IDs in {target_release} ---")

for sae_id, hook_point in directory[target_release].saes_map.items():
    print(f"ID: {sae_id} (Hooks into: {hook_point})")

sae_id = "layer_17_width_16k_l0_big"

print(f"Loading SAE: {sae_id}...")
sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
    release=target_release,
    sae_id=sae_id,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="bfloat16"
)
print("SAE loaded successfully!")
print(cfg_dict)
print(sparsity)

# View the type of the object
print(f"Class: {type(sae)}")

# Print the PyTorch module architecture
print("\n--- SAE Architecture ---")
print(sae)

norm_strategy = sae.cfg.normalize_activations

if norm_strategy in ["none", "expected_average_only_in"]:
    print(f"Normalization Out: EMPTY (Strategy is '{norm_strategy}')")
else:
    print(f"Normalization Out: ACTIVE (Strategy is '{norm_strategy}')")

reshape_strategy = sae.cfg.reshape_activations

if reshape_strategy == "none":
    print("Reshape Out: EMPTY (Identity function)")
else:
    print(f"Reshape Out: ACTIVE (Strategy is '{reshape_strategy}')")

import torch
from transformer_lens import HookedTransformer

# 1. Specify the exact model ID from Hugging Face
model_id = "google/gemma-3-270m" # or "google/gemma-3-270m-it"

print(f"Loading {model_id}...")

# 2. Load the HookedTransformer
# We use bfloat16 because Gemma models are natively trained in that dtype
model = HookedSAETransformer.from_pretrained_no_processing(
    model_id,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16
)

print("Model loaded successfully!")

# 3. Run a prompt and cache the internal states
prompt = "Artificial intelligence will eventually"
logits, cache = model.run_with_cache_with_saes(prompt, saes = [sae])
print(cache)

# 4. Extract the exact layer activations for your SAE
# TransformerLens uses a specific naming convention for its hook points.
# For example, to get the residual stream output at layer 12:
#layer_12_acts = cache["blocks.12.hook_resid_post"]

#print(f"Shape of Layer 12 activations: {layer_12_acts.shape}")

# (Optional but recommended) Cast SAE to match the model's precision
# to ensure perfectly identical mathematical operations
sae = sae.to(dtype=torch.bfloat16)

model_id = "google/gemma-3-270m"
print(f"Loading {model_id} via NNsight...")
model = LanguageModel(model_id, device_map="cuda", torch_dtype=torch.bfloat16)

with model.trace(prompt):
    hidden_states = model.model.layers[17].output[0].save()
    feature_acts = sae.encode(hidden_states)
    saved_feature_acts = feature_acts.save()

# 1. Unpack the NNsight wrapper using .value
nnsight_acts = saved_feature_acts.value

# 2. Extract the post-thresholding activations from TransformerLens
tlens_acts = cache["blocks.17.hook_resid_post.hook_sae_acts_post"]

# Extract the hidden states directly from both base models
nnsight_hidden = hidden_states.value
tlens_hidden = cache["blocks.17.hook_resid_post.hook_sae_input"]

print(nnsight_hidden)
print(tlens_hidden)
# Measure the maximum absolute difference between the base models
max_hidden_diff = (nnsight_hidden - tlens_hidden).abs().max()
print(f"Max difference in base LM hidden states: {max_hidden_diff:.4f}")

# Manually pass the NNsight hidden state into the SAELens encode function
manual_nnsight_encode = sae.encode(nnsight_hidden)
max_sae_diff = (nnsight_acts - manual_nnsight_encode).abs().max()
print(f"Max difference if inputs were identical: {max_sae_diff:.4f}")

# 3. Add a small tolerance for bfloat16 mathematical operations
assert torch.allclose(nnsight_acts, tlens_acts, atol=1e-3, rtol=1e-3), "Activations do not match!"
print("Assertion passed successfully! The feature activations are identical.")