from nnsight import NNsight
import numpy as np
from tqdm import tqdm

def get_target_layer_index(edge_str):
    """Parses edge string to find the target layer index for sorting."""
    _, v = edge_str.split("->")
    if v == "logits": return 999
    # Format usually 'a10.h2' or 'm5'
    if v.startswith("a"): return 2*int(v.split(".")[0][1:])
    if v.startswith("m"): return 2*int(v[1:])+1
    return -1


def patch_activations(model, edges_to_patch, clean_input, corrupted_input, target, metric, batch_size = 8):
    model_nnsight = NNsight(model)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    edges_sorted = sorted(edges_to_patch, key=get_target_layer_index)

    num_batches = (len(clean_input) + batch_size - 1) // batch_size
    metric_cache = []

    for batch_idx in tqdm(range(num_batches), desc="patch batches"):
        # 1. PRE-COMPUTE CLEAN VALUES (Static Targets)
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        input_cl = clean_input[batch_start:batch_end]
        input_cor = corrupted_input[batch_start:batch_end]
        target_batch = target[batch_start:batch_end]

        clean_values = {}
        with model_nnsight.trace(input_cl):
            clean_values["input"] = model_nnsight.vit.embeddings.output.save()

            for layer_idx in range(n_layers):
                layer_module = model_nnsight.vit.encoder.layer[layer_idx]

                # MLP Output
                clean_values[f"m{layer_idx}"] = layer_module.output.dense.output.save()

                # Attention Heads (Projected)
                # We need the output of the SelfAttention (before the dense Output layer)
                # AND the weights of the Dense Output layer to manually project later.
                concat_heads = layer_module.attention.attention.output[0]
                W_O = layer_module.attention.output.dense.weight

                for h in range(n_heads):
                    start, end = h * head_dim, (h + 1) * head_dim
                    # Project specific head
                    head_z = concat_heads[:, :, start:end]
                    W_O_head = W_O[:, start:end]
                    clean_values[f"a{layer_idx}.h{h}"] = (head_z @ W_O_head.T).save()

        # 2. ITERATIVE PATCHING (Dynamic Steering)
        patched_logits = None
        patched_loss = None

        with model_nnsight.trace(input_cor) as tracer:

            # We iterate through the edges.
            # Since nnsight builds a graph, the order of definition here matters.
            # Upstream patches defined first will affect downstream nodes naturally.
            resids_pre_attention = {}
            for edge in edges_sorted:
                u, v = edge.split("->")

                # --- A. GET LIVE VALUE OF SOURCE (u) ---
                # This is the key fix: We don't use a saved dictionary.
                # We compute the value of 'u' right now in the graph.

                current_u_value = None

                if u == "input":
                    current_u_value = model_nnsight.vit.embeddings.output
                elif u.startswith("m"):
                    l_src = int(u[1:])
                    # Live output of the MLP
                    current_u_value = model_nnsight.vit.encoder.layer[l_src].output.dense.output
                elif u.startswith("a"):
                    l_src = int(u.split(".")[0][1:])
                    h_src = int(u.split(".")[1][1:])

                    # Re-construct the specific head's output from the live stream
                    layer_module = model_nnsight.vit.encoder.layer[l_src]
                    concat_heads = layer_module.attention.attention.output[0]
                    W_O = layer_module.attention.output.dense.weight

                    start, end = h_src * head_dim, (h_src + 1) * head_dim

                    head_z = concat_heads[:, :, start:end]
                    W_O_head = W_O[:, start:end]
                    current_u_value = head_z @ W_O_head.T

                # --- B. CALCULATE STEERING ---
                # Steering = Target (Clean) - Current (Potentially modified by upstream patches)
                steering = clean_values[u] - current_u_value

                # --- B.2 CACHE THE RESID CACHES ---
                # This is useful when changing multiple incoming edges
                # to a single attention head.
                u_layer = None
                if u == "input": u_layer = 0
                elif u.startswith("a"): u_layer = int(u.split(".")[0][1:]) + 1
                elif u.startswith("m"): u_layer = int(u[1:]) + 1

                v_layer = None
                if v == "logits": v_layer = n_layers
                if v.startswith("a"): v_layer = int(v.split(".")[0][1:])
                if v.startswith("m"): v_layer = int(v[1:])

                # --- C. APPLY PATCH TO DESTINATION (v) ---
                if v == "logits":
                    model_nnsight.vit.layernorm.input += steering

                elif v.startswith("a"):
                    l_tgt = int(v.split(".")[0][1:])
                    h_tgt = int(v.split(".")[1][1:-3])
                    head_type = v.split(".")[1][-3:]

                    layer_module = model_nnsight.vit.encoder.layer[l_tgt]
                    ln_module = layer_module.layernorm_before

                    # Through-LN Logic
                    current_resid_input = None
                    if (v in resids_pre_attention):
                        current_resid_input = resids_pre_attention[v]
                    else:
                        current_resid_input = ln_module.input
                    corrected_post_ln = ln_module(current_resid_input + steering)
                    current_post_ln = ln_module(current_resid_input)  # OR ln_module.output if available

                    delta_post_ln = corrected_post_ln - current_post_ln

                    # Project and inject
                    start = h_tgt * head_dim
                    end = (h_tgt + 1) * head_dim

                    if head_type == "<q>":
                        W_Q = layer_module.attention.attention.query.weight
                        delta_q = delta_post_ln @ W_Q.T
                        layer_module.attention.attention.query.output[:, :, start:end] += delta_q[:, :, start:end]
                    if head_type == "<k>":
                        W_K = layer_module.attention.attention.key.weight
                        delta_k = delta_post_ln @ W_K.T
                        layer_module.attention.attention.key.output[:, :, start:end] += delta_k[:, :, start:end]
                    if head_type == "<v>":
                        W_V = layer_module.attention.attention.value.weight
                        delta_v = delta_post_ln @ W_V.T
                        layer_module.attention.attention.value.output[:, :, start:end] += delta_v[:, :, start:end]

                elif v.startswith("m"):
                    l_tgt = int(v[1:])
                    model_nnsight.vit.encoder.layer[l_tgt].layernorm_after.input += steering

                if (v not in resids_pre_attention):
                    if v_layer < n_layers:
                        resids_pre_attention[v] = model_nnsight.vit.encoder.layer[v_layer].layernorm_before.input + steering
                    else:
                        resids_pre_attention[v] = model_nnsight.vit.layernorm.input + steering
                else:
                    resids_pre_attention[v] = resids_pre_attention[v] + steering

            patched_logits = model_nnsight.classifier.output

            # --- D. METRICS ---
            patched_loss = metric(patched_logits, target_batch).save()
        metric_cache.append(patched_loss.value.item())

    return np.mean(metric_cache), np.array(metric_cache)

