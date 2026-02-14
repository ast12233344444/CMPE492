import torch
from nnsight import NNsight
from torch.nn import functional as F
from Attribute import backprop_through_layernorm_functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inverse_sigmoid(x, eps=1e-6):
    """Converts [0, 1] tensor to logit space (-inf, inf)."""
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def maximize_edge_effect_on_metric(model, clean_input_normalized, edge, targets,
                                   normalization_mean, normalization_std,
                                   metric=F.cross_entropy,
                                   iterations=100, increase=True):
    u, v = edge.split("->")
    model_nnsight = NNsight(model)
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    # Move stats to device for math
    mean = torch.tensor(normalization_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(normalization_std, device=device).view(1, -1, 1, 1)

    # --- FIX 1: DENORMALIZE INITIALIZATION ---
    # Convert the normalized input (-2, 2) back to (0, 1) before inverse_sigmoid
    clean_input_01 = clean_input_normalized * std + mean
    input_logits = inverse_sigmoid(clean_input_01)

    input_param = torch.nn.Parameter(input_logits)
    optimizer = torch.optim.Adam([input_param], lr=0.05)

    for i in range(iterations):
        optimizer.zero_grad()

        # 1. Get image in [0, 1] space
        valid_image_01 = torch.sigmoid(input_param)
        model_input = (valid_image_01 - mean) / std

        target_grad_v_val = None
        ln_module_ref = None
        pre_ln_input_val = None

        # --- PASS 1: Get the Sensitivity (Gradient at v) ---
        # Detach model_input so we don't backprop metric loss to input
        with model_nnsight.trace(model_input.detach()) as tracer:
            if v == "logits":
                grad_v_proxy = model_nnsight.vit.layernorm.input.grad.save()

            elif v.startswith("m"):
                layer_idx = int(v[1:])
                grad_v_proxy = model_nnsight.vit.encoder.layer[layer_idx].layernorm_after.input.grad.save()

            elif v.startswith("a"):
                head_name, token_type = v[:-3], v[-3:]
                layer_i, head_i = head_name.split(".")
                layer_idx = int(layer_i[1:])
                head_idx = int(head_i[1:])
                layer_module = model_nnsight.vit.encoder.layer[layer_idx]

                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim

                if token_type == "<q>":
                    grads = layer_module.attention.attention.query.output.grad
                    weights = layer_module.attention.attention.query.weight
                elif token_type == "<k>":
                    grads = layer_module.attention.attention.key.output.grad
                    weights = layer_module.attention.attention.key.weight
                elif token_type == "<v>":
                    grads = layer_module.attention.attention.value.output.grad
                    weights = layer_module.attention.attention.value.weight

                grad_projected = grads[:, :, start:end] @ weights[start:end, :]
                grad_v_proxy = grad_projected.save()

                ln_module_ref = layer_module.layernorm_before
                pre_ln_input_val = ln_module_ref.input.save()

            logits = model_nnsight.classifier.output
            loss_val = metric(logits, targets)
            loss_val.backward()

        # --- Post-Processing Pass 1 ---
        grad_v_val = grad_v_proxy.value

        if v.startswith("a"):
            target_grad_v_val = backprop_through_layernorm_functional(grad_v_val, pre_ln_input_val, ln_module_ref)
        else:
            target_grad_v_val = grad_v_val

        target_grad_v_val = target_grad_v_val.detach()

        # --- PASS 2: Maximize Attribution ---
        # We trace the model_input (which is connected to input_param via sigmoid)
        with model_nnsight.trace(model_input) as tracer:
            if u == "input":
                node_u = model_nnsight.vit.embeddings.output.save()
            elif u.startswith("m"):
                layer_idx = int(u[1:])
                node_u = model_nnsight.vit.encoder.layer[layer_idx].output.dense.output.save()
            elif u.startswith("a"):
                # ... (Same parsing logic as before) ...
                layer_i, head_i = u.split(".")
                layer_idx = int(layer_i[1:])
                head_idx = int(head_i[1:])
                layer_module = model_nnsight.vit.encoder.layer[layer_idx]

                concat_heads = layer_module.attention.attention.output[0]
                W_O = layer_module.attention.output.dense.weight

                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim

                head_z = concat_heads[:, :, start:end]
                W_O_head = W_O[:, start:end]

                node_u = (head_z @ W_O_head.T).save()

            # EAP Objective
            grad_tensor = torch.tensor(target_grad_v_val, device=device)
            edge_attr = (node_u * grad_tensor).sum(dim=-1).mean()

            if increase:
                loss = -edge_attr
            else:
                loss = edge_attr
            loss.backward()

        # 3. Step Optimizer
        optimizer.step()

    # Return the valid [0, 1] image, not the logits!
    return torch.sigmoid(input_param).detach()