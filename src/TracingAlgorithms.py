import os
from sklearn.decomposition import PCA
import numpy as np
import torch
from matplotlib import pyplot as plt
from nnsight import NNsight
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TracingAlgorithms:

    @staticmethod
    def _backprop_through_layernorm_functional(grad_post_ln, resid_input, ln_module):
        x = resid_input.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            output = ln_module(x)

        grads = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_post_ln,
            retain_graph=False,
            create_graph=False
        )
        return grads[0]

    @staticmethod
    def _get_target_layer_index(edge_str):
        """Parses edge string to find the target layer index for sorting."""
        _, v = edge_str.split("->")
        if v == "logits": return 999
        # Format usually 'a10.h2' or 'm5'
        if v.startswith("a"): return 2 * int(v.split(".")[0][1:])
        if v.startswith("m"): return 2 * int(v[1:]) + 1
        return -1

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-6):
        """Converts [0, 1] tensor to logit space (-inf, inf)."""
        x = x.clamp(eps, 1 - eps)
        return torch.log(x / (1 - x))

    @staticmethod
    def _get_activations(model_ref, u, head_dim):
        if u == "input":
            clean_activations = model_ref.vit.embeddings.output.save()
        elif u.startswith("m"):
            layer_idx = int(u[1:])
            layer_module = model_ref.vit.encoder.layer[layer_idx]
            clean_activations = layer_module.output.dense.output.save()
        elif u.startswith("a"):
            layer, head = u.split(".")
            layer_idx = int(layer[1:])
            head_idx = int(head[1:])
            layer_module = model_ref.vit.encoder.layer[layer_idx]
            concat_heads = layer_module.attention.attention.output[0]
            W_O = layer_module.attention.output.dense.weight
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            head_z = concat_heads[:, :, start:end]
            W_O_head = W_O[:, start:end]
            clean_activations = (head_z @ W_O_head.T).save()
        return clean_activations

    @staticmethod
    def EAP(model_nnsight, graph, input_clean, input_adversarial, truth_label=None,
            batch_size=8, invert=False, img_path="pruned_compgraph.png",
            metric_fn = F.cross_entropy):
        n_layers = model_nnsight.config.num_hidden_layers
        n_heads = model_nnsight.config.num_attention_heads
        head_dim = model_nnsight.config.hidden_size // n_heads

        edge_accumulators = {edge: 0.0 for edge in graph.edges}
        n_total = input_clean.shape[0]

        num_batches = (n_total + batch_size - 1) // batch_size
        print(f"Processing {n_total} samples in {num_batches} batches (Batch size: {batch_size})...")

        for batch_idx in tqdm(range(num_batches), f"processing batches..."):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_total)

            current_clean = input_clean[start:end]
            current_adv = input_adversarial[start:end]

            clean_out = {}
            corrupted_out = {}
            clean_in_grad = {}
            with model_nnsight.trace() as tracer:

                with tracer.invoke(current_clean) as invoker_clean:
                    clean_out["input"] = model_nnsight.vit.embeddings.output.save()

                    for layer_idx in range(n_layers):
                        layer_module = model_nnsight.vit.encoder.layer[layer_idx]
                        concat_heads = layer_module.attention.attention.output[0]
                        W_O = layer_module.attention.output.dense.weight

                        clean_out[f"resid_pre_ln_{layer_idx}"] = layer_module.layernorm_before.input.save()

                        Q_grads = layer_module.attention.attention.query.output.grad
                        W_Q = layer_module.attention.attention.query.weight
                        K_grads = layer_module.attention.attention.key.output.grad
                        W_K = layer_module.attention.attention.key.weight
                        V_grads = layer_module.attention.attention.value.output.grad
                        W_V = layer_module.attention.attention.value.weight

                        for head_idx in range(n_heads):
                            start = head_idx * head_dim
                            end = (head_idx + 1) * head_dim
                            head_z = concat_heads[:, :, start:end]
                            W_O_head = W_O[:, start:end]
                            clean_out[f"a{layer_idx}.h{head_idx}"] = (head_z @ W_O_head.T).save()

                            clean_in_grad[f"a{layer_idx}.h{head_idx}<q>"] = (Q_grads[:, :, start:end] @ W_Q[start:end, :]).save()
                            clean_in_grad[f"a{layer_idx}.h{head_idx}<k>"] = (K_grads[:, :, start:end] @ W_K[start:end, :]).save()
                            clean_in_grad[f"a{layer_idx}.h{head_idx}<v>"] = (V_grads[:, :, start:end] @ W_V[start:end, :]).save()

                        clean_in_grad[f"m{layer_idx}"] = layer_module.layernorm_after.input.grad.save()
                        clean_out[f"m{layer_idx}"] = layer_module.output.dense.output.save()

                    clean_in_grad["logits"] = model_nnsight.vit.layernorm.input.grad.save()

                    logits = model_nnsight.classifier.output.save()

                    metric = metric_fn(logits, torch.tensor([truth_label for _ in range(batch_size)], device=device))

                    metric.backward()

                with tracer.invoke(current_adv) as invoker_adversarial:
                    corrupted_out["input"] = model_nnsight.vit.embeddings.output.save()

                    for layer_idx in range(n_layers):
                        layer_module = model_nnsight.vit.encoder.layer[layer_idx]
                        concat_heads = layer_module.attention.attention.output[0]
                        W_O = layer_module.attention.output.dense.weight

                        for head_idx in range(n_heads):
                            start = head_idx * head_dim
                            end = (head_idx + 1) * head_dim
                            head_z = concat_heads[:, :, start:end]
                            W_O_head = W_O[:, start:end]
                            corrupted_out[f"a{layer_idx}.h{head_idx}"] = (head_z @ W_O_head.T).save()

                        corrupted_out[f"m{layer_idx}"] = layer_module.output.dense.output.save()

            for key in clean_out:
                clean_out[key] = clean_out[key].value
            for key in clean_in_grad:
                clean_in_grad[key] = clean_in_grad[key].value
            for key in corrupted_out:
                corrupted_out[key] = corrupted_out[key].value

            for layer_idx in range(n_layers):
                layer_input = clean_out[f"resid_pre_ln_{layer_idx}"]
                ln_module = model_nnsight.vit.encoder.layer[layer_idx].layernorm_before
                for head_idx in range(n_heads):
                    clean_in_grad[f"a{layer_idx}.h{head_idx}<q>"] = TracingAlgorithms._backprop_through_layernorm_functional(
                        clean_in_grad[f"a{layer_idx}.h{head_idx}<q>"], layer_input, ln_module)
                    clean_in_grad[f"a{layer_idx}.h{head_idx}<k>"] = TracingAlgorithms._backprop_through_layernorm_functional(
                        clean_in_grad[f"a{layer_idx}.h{head_idx}<k>"], layer_input, ln_module)
                    clean_in_grad[f"a{layer_idx}.h{head_idx}<v>"] = TracingAlgorithms._backprop_through_layernorm_functional(
                        clean_in_grad[f"a{layer_idx}.h{head_idx}<v>"], layer_input, ln_module)

            for edge in graph.edges:
                nodes = edge.split("->")
                source = nodes[0]
                target = nodes[1]

                score = None
                if clean_in_grad[target].dim() == 2: # target is logits
                    score = torch.einsum("b c d, b d -> b c", corrupted_out[source] - clean_out[source], clean_in_grad[target])
                    score = score.sum(dim=-1)
                    score = score.sum()
                else:
                    score = torch.einsum("b c d, b c d-> b c", corrupted_out[source] - clean_out[source], clean_in_grad[target])
                    score = score.sum(dim=-1)
                    score = score.sum()
                if not invert:
                    edge_accumulators[edge] += score.item()
                else:
                    edge_accumulators[edge] -= score.item()

        scores =[]
        for edge in graph.edges:
            scores.append( edge_accumulators[edge] / n_total )
            graph.edges[edge].score = edge_accumulators[edge] / n_total

        graph.reset(False)
        graph.apply_topn(50, absolute=False, reset=True, prune=False)

        graph.to_image(img_path)

    @staticmethod
    def patch_activations(model_nnsight, edges_to_patch, clean_input, corrupted_input, target, metric, batch_size = 8):
        #CALCULTAES RESULTS BY CLEANING THE CORRUPTED OUTPUTS
        n_layers = model_nnsight.config.num_hidden_layers
        n_heads = model_nnsight.config.num_attention_heads
        head_dim = model_nnsight.config.hidden_size // n_heads

        edges_sorted = sorted(edges_to_patch, key=TracingAlgorithms._get_target_layer_index)

        num_batches = (len(clean_input) + batch_size - 1) // batch_size
        metric_cache = []

        for batch_idx in range(num_batches):
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

                    clean_values[f"m{layer_idx}"] = layer_module.output.dense.output.save()

                    concat_heads = layer_module.attention.attention.output[0]
                    W_O = layer_module.attention.output.dense.weight

                    for h in range(n_heads):
                        start, end = h * head_dim, (h + 1) * head_dim
                        head_z = concat_heads[:, :, start:end]
                        W_O_head = W_O[:, start:end]
                        clean_values[f"a{layer_idx}.h{h}"] = (head_z @ W_O_head.T).save()

            patched_logits = None
            patched_loss = None

            with model_nnsight.trace(input_cor) as tracer:

                resids_pre_attention = {}
                for edge in edges_sorted:
                    u, v = edge.split("->")

                    current_u_value = None

                    if u == "input":
                        current_u_value = model_nnsight.vit.embeddings.output
                    elif u.startswith("m"):
                        l_src = int(u[1:])
                        current_u_value = model_nnsight.vit.encoder.layer[l_src].output.dense.output
                    elif u.startswith("a"):
                        l_src = int(u.split(".")[0][1:])
                        h_src = int(u.split(".")[1][1:])

                        layer_module = model_nnsight.vit.encoder.layer[l_src]
                        concat_heads = layer_module.attention.attention.output[0]
                        W_O = layer_module.attention.output.dense.weight

                        start, end = h_src * head_dim, (h_src + 1) * head_dim

                        head_z = concat_heads[:, :, start:end]
                        W_O_head = W_O[:, start:end]
                        current_u_value = head_z @ W_O_head.T

                    steering = clean_values[u] - current_u_value

                    u_layer = None
                    if u == "input": u_layer = 0
                    elif u.startswith("a"): u_layer = int(u.split(".")[0][1:]) + 1
                    elif u.startswith("m"): u_layer = int(u[1:]) + 1

                    v_layer = None
                    if v == "logits": v_layer = n_layers
                    if v.startswith("a"): v_layer = int(v.split(".")[0][1:])
                    if v.startswith("m"): v_layer = int(v[1:])

                    if v == "logits":
                        model_nnsight.vit.layernorm.input += steering

                    elif v.startswith("a"):
                        l_tgt = int(v.split(".")[0][1:])
                        h_tgt = int(v.split(".")[1][1:-3])
                        head_type = v.split(".")[1][-3:]

                        layer_module = model_nnsight.vit.encoder.layer[l_tgt]
                        ln_module = layer_module.layernorm_before

                        current_resid_input = None
                        if (v in resids_pre_attention):
                            current_resid_input = resids_pre_attention[v]
                        else:
                            current_resid_input = ln_module.input
                        corrected_post_ln = ln_module(current_resid_input + steering)
                        current_post_ln = ln_module(current_resid_input)  # OR ln_module.output if available

                        delta_post_ln = corrected_post_ln - current_post_ln

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

                patched_loss = metric(patched_logits, target_batch).save()
            metric_cache.append(patched_loss.value.item())

        return np.mean(metric_cache), np.array(metric_cache)

    @staticmethod
    def maximize_edge_effect_on_metric(model, clean_input, edge, targets,
                                       normalization_mean, normalization_std,
                                       metric=F.cross_entropy,
                                       iterations=100, increase=True):
        u, v = edge.split("->")
        model_nnsight = NNsight(model)
        n_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // n_heads

        mean = torch.tensor(normalization_mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(normalization_std, device=device).view(1, -1, 1, 1)

        clean_input_01 = clean_input * std + mean
        input_logits = TracingAlgorithms._inverse_sigmoid(clean_input_01)

        input_param = torch.nn.Parameter(input_logits)
        optimizer = torch.optim.Adam([input_param], lr=0.05)

        for i in range(iterations):
            optimizer.zero_grad()

            valid_image_01 = torch.sigmoid(input_param)
            model_input = (valid_image_01 - mean) / std

            target_grad_v_val = None
            ln_module_ref = None
            pre_ln_input_val = None

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

            grad_v_val = grad_v_proxy.value

            if v.startswith("a"):
                target_grad_v_val = TracingAlgorithms._backprop_through_layernorm_functional(grad_v_val, pre_ln_input_val, ln_module_ref)
            else:
                target_grad_v_val = grad_v_val

            target_grad_v_val = target_grad_v_val.detach()

            with model_nnsight.trace(model_input) as tracer:
                if u == "input":
                    node_u = model_nnsight.vit.embeddings.output.save()
                elif u.startswith("m"):
                    layer_idx = int(u[1:])
                    node_u = model_nnsight.vit.encoder.layer[layer_idx].output.dense.output.save()
                elif u.startswith("a"):
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

                grad_tensor = torch.tensor(target_grad_v_val, device=device)
                edge_attr = (node_u * grad_tensor).sum(dim=-1).mean()

                if increase:
                    loss = -edge_attr
                else:
                    loss = edge_attr
                loss.backward()

            optimizer.step()

        return torch.sigmoid(input_param).detach()

    @staticmethod
    def trace_contrib_integrated_gradients(model_nnsight, clean_input, corrupted_input,
                                           edge_to_measure, resolution, metric, target):
        # CALCULATES RESULTS BY CLEANING THE CORRUPT
        save_dir = f"/home/ahmet/PycharmProjects/CMPE492/results/{edge_to_measure}_{resolution}_igtrace.png"

        def steer_inputs(model_ref, v, steering):
            if v == "logits":
                model_ref.vit.layernorm.input += steering

            elif v.startswith("a"):
                l_tgt = int(v.split(".")[0][1:])
                h_tgt = int(v.split(".")[1][1:-3])
                head_type = v.split(".")[1][-3:]

                layer_module = model_ref.vit.encoder.layer[l_tgt]
                ln_module = layer_module.layernorm_before

                current_resid_input = ln_module.input
                corrected_post_ln = ln_module(current_resid_input + steering)
                current_post_ln = ln_module(current_resid_input)

                delta_post_ln = corrected_post_ln - current_post_ln

                start = h_tgt * head_dim
                end = (h_tgt + 1) * head_dim

                def replace_slice(output_tensor, delta, start, end):
                    parts = []
                    if start > 0:
                        parts.append(output_tensor[..., :start])

                    parts.append(output_tensor[..., start:end] + delta[..., start:end])

                    if end < model_ref.config.hidden_size:
                        parts.append(output_tensor[..., end:])

                    return torch.cat(parts, dim=-1)

                if head_type == "<q>":
                    W_Q = layer_module.attention.attention.query.weight
                    delta_q = delta_post_ln @ W_Q.T
                    layer_module.attention.attention.query.output = replace_slice(
                    layer_module.attention.attention.query.output, delta_q, start, end)
                if head_type == "<k>":
                    W_K = layer_module.attention.attention.key.weight
                    delta_k = delta_post_ln @ W_K.T
                    layer_module.attention.attention.key.output = replace_slice(
                        layer_module.attention.attention.key.output, delta_k, start, end)
                if head_type == "<v>":
                    W_V = layer_module.attention.attention.value.weight
                    delta_v = delta_post_ln @ W_V.T
                    layer_module.attention.attention.value.output = replace_slice(
                        layer_module.attention.attention.value.output, delta_v, start, end)

            elif v.startswith("m"):
                l_tgt = int(v[1:])
                model_ref.vit.encoder.layer[l_tgt].layernorm_after.input += steering

        def get_gradients(model_ref, v):
            if v == "logits":
                grad_v_proxy = model_ref.vit.layernorm.input.grad.save()
                return  grad_v_proxy

            elif v.startswith("m"):
                layer_idx = int(v[1:])
                grad_v_proxy = model_ref.vit.encoder.layer[layer_idx].layernorm_after.input.grad.save()
                return grad_v_proxy

            elif v.startswith("a"):
                head_name, token_type = v[:-3], v[-3:]
                layer_i, head_i = head_name.split(".")
                layer_idx = int(layer_i[1:])
                head_idx = int(head_i[1:])
                layer_module = model_ref.vit.encoder.layer[layer_idx]

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

                return grad_v_proxy, ln_module_ref, pre_ln_input_val

        n_layers = model_nnsight.config.num_hidden_layers
        n_heads = model_nnsight.config.num_attention_heads
        head_dim = model_nnsight.config.hidden_size // n_heads
        u, v = edge_to_measure.split("->")

        clean_activation = None
        corrupted_activation = None
        with model_nnsight.trace() as tracer:
            with tracer.invoke(clean_input) as invoker_clean:
                clean_activation = TracingAlgorithms._get_activations(model_nnsight, u, head_dim)

            with tracer.invoke(corrupted_input) as invoker_corrupted:
                corrupted_activation = TracingAlgorithms._get_activations(model_nnsight, u, head_dim)

        clean_activation = clean_activation.value.detach()
        corrupted_activation = corrupted_activation.value.detach()

        metric_vals = []
        grad_measurement_sizes = []
        cosine_similarities = []

        for i in tqdm(range(resolution+1), "interpolating..."):
            current_activation = corrupted_activation * (resolution-i)/resolution + clean_activation * i / resolution
            current_steering = current_activation - corrupted_activation
            with model_nnsight.trace() as tracer:
                with tracer.invoke(corrupted_input) as invoker_corrupted:
                    steer_inputs(model_nnsight, v, current_steering)
                    if v.startswith('a'):
                        grad_v_proxy, ln_module_ref, pre_ln_input_val = get_gradients(model_nnsight, v)
                    else:
                        grad_v_proxy = get_gradients(model_nnsight, v)
                    patched_logits = model_nnsight.classifier.output

                    patched_loss = metric(patched_logits, target).save()
                    patched_loss.backward()

            metric_vals.append(patched_loss.value.cpu().detach().numpy())
            if v.startswith('a'):
                grad_tensor = TracingAlgorithms._backprop_through_layernorm_functional(
                        grad_v_proxy.value, pre_ln_input_val, ln_module_ref)
            else:
                grad_tensor = grad_v_proxy.value

            grad_tensor_norm = torch.linalg.vector_norm(grad_tensor, ord=2, dim=-1).mean().cpu().detach().numpy()
            current_steering_norm = torch.linalg.vector_norm(current_steering, ord=2, dim=-1).mean().cpu().detach().numpy()
            grad_measurement_sizes.append(grad_tensor_norm)
            cosine_similarities.append((grad_tensor * current_steering).sum(dim = -1).mean().cpu().detach().numpy() / (grad_tensor_norm * current_steering_norm))

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        steps = range(len(metric_vals))

        axes[0].plot(steps, metric_vals, color='tab:blue', marker='o', markersize=3)
        axes[0].set_ylabel("Metric Value (Loss)")
        axes[0].set_title(f"IG Trace: {edge_to_measure}")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, grad_measurement_sizes, color='tab:orange', marker='o', markersize=3)
        axes[1].set_ylabel("Gradient L2 Norm")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, cosine_similarities, color='tab:green', marker='o', markersize=3)
        axes[2].set_ylabel("Grad-Steering Similarity")
        axes[2].set_xlabel("Resolution Step")
        axes[2].grid(True, alpha=0.3)

        save_path = save_dir

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Trace plot saved to: {save_path}")

    @staticmethod
    def trace_activation_patterns(model_nnsight, edge, clean_input_a, corrupted_input, clean_input_b, batch_size = 8, reduce_mean=True):
        n_layers = model_nnsight.config.num_hidden_layers
        n_heads = model_nnsight.config.num_attention_heads
        head_dim = model_nnsight.config.hidden_size // n_heads

        u, v = edge.split("->")
        n_samples = clean_input_a.shape[0]
        n_batches = (batch_size + n_samples - 1) // batch_size

        clean_a_outputs = []
        clean_b_outputs = []
        corrupted_outputs = []


        for batch_idx in tqdm(range(n_batches), "iterating batches..."):
            begin = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size

            clean_a_batch = clean_input_a[begin:end]
            clean_b_batch = clean_input_b[begin:end]
            corrupted_batch = corrupted_input[begin:end]

            with model_nnsight.trace() as tracer:
                with tracer.invoke(clean_a_batch) as _:
                    outputs = TracingAlgorithms._get_activations(model_nnsight, u, head_dim)
            if reduce_mean:
                outputs = torch.mean(outputs, dim=1)
            clean_a_outputs.append(outputs.view(-1, outputs.shape[-1]).detach().cpu().numpy())

            with model_nnsight.trace() as tracer:
                with tracer.invoke(clean_b_batch) as _:
                    outputs = TracingAlgorithms._get_activations(model_nnsight, u, head_dim)
            if reduce_mean:
                outputs = torch.mean(outputs, dim=1)
            clean_b_outputs.append(outputs.view(-1, outputs.shape[-1]).detach().cpu().numpy())

            with model_nnsight.trace() as tracer:
                with tracer.invoke(corrupted_batch) as _:
                    outputs = TracingAlgorithms._get_activations(model_nnsight, u, head_dim)
            if reduce_mean:
                outputs = torch.mean(outputs, dim = 1)
            corrupted_outputs.append(outputs.view(-1, outputs.shape[-1]).detach().cpu().numpy())

        data_a = np.concatenate(clean_a_outputs, axis=0)
        data_b = np.concatenate(clean_b_outputs, axis=0)
        data_corr = np.concatenate(corrupted_outputs, axis=0)

        labels_a = np.zeros(len(data_a))
        labels_b = np.ones(len(data_b))
        labels_corr = np.full(len(data_corr), 2)

        combined_labels = np.concatenate([labels_a, labels_b, labels_corr], axis=0)

        combined_data = np.concatenate([data_a, data_b, data_corr], axis=0)

        lda = LinearDiscriminantAnalysis(n_components=2)
        combined_2d = lda.fit_transform(combined_data, combined_labels)

        len_a = len(data_a)
        len_b = len(data_b)

        a_2d = combined_2d[:len_a]
        b_2d = combined_2d[len_a: len_a + len_b]
        corr_2d = combined_2d[len_a + len_b:]

        plt.figure(figsize=(10, 8))
        alpha = 0.6  # Transparency helps visualize density overlap
        s = 15  # Point size

        plt.scatter(a_2d[:, 0], a_2d[:, 1], c='blue', label='Clean A', alpha=alpha, s=s)
        plt.scatter(b_2d[:, 0], b_2d[:, 1], c='green', label='Clean B', alpha=alpha, s=s)
        plt.scatter(corr_2d[:, 0], corr_2d[:, 1], c='red', label='Corrupted', alpha=alpha, s=s)

        plt.title(f"Activation Pattern Projection: {u} -> {v}")
        plt.xlabel(f"PC1 (Explained Var: {lda.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 (Explained Var: {lda.explained_variance_ratio_[1]:.2%})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        filename = f"results/trace_scatter_act_{u}_lda{"_reduce_mean" if reduce_mean else ""}.png"
        plt.savefig(filename)
        plt.close()  # Close memory reference

        print(f"Scatterplot saved to {filename}")


