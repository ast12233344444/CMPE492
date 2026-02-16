import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification
from nnsight import NNsight
from PIL import Image
from torch.nn import functional as F

import ViTCompGraph
from patching import patch_activations
import edge_effect_isolation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO List
#   Activation Patching to see how much the loss decreases Done
#   EAP-IG
#   Edge Corruption Maximization Done
#   Edge Significance (consistency) Done


def backprop_through_layernorm_functional(grad_post_ln, resid_input, ln_module):
    # 1. Detach and require grad
    x = resid_input.clone().detach().requires_grad_(True)

    # 2. Forward pass
    with torch.enable_grad():
        output = ln_module(x)

    # 3. Functional Backward pass
    # outputs: The result of the forward pass
    # inputs: The tensor(s) you want gradients for
    # grad_outputs: The 'seed' gradient (your grad_post_ln)
    grads = torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=grad_post_ln,
        retain_graph=False,
        create_graph=False
    )

    # grads is a tuple (grad_x,), so we take the first element
    return grads[0]

def EAP(model, graph, input_clean, input_adversarial, truth_label=None,
        batch_size=8, invert=False, img_path="pruned_compgraph.png",
        metric_fn = F.cross_entropy):
    model_nnsight = NNsight(model)

    n_layers = 12
    n_heads = 12
    d_model = 768
    head_dim = d_model // n_heads

    edge_accumulators = {edge: 0.0 for edge in graph.edges}
    n_total = input_clean.shape[0]

    num_batches = (n_total + batch_size - 1) // batch_size
    print(f"Processing {n_total} samples in {num_batches} batches (Batch size: {batch_size})...")

    for batch_idx in tqdm(range(num_batches), f"processing batches..."):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_total)

        current_clean = input_clean[start:end]
        current_adv = input_adversarial[start:end]

        # Local storage for this batch
        clean_out = {}
        corrupted_out = {}
        clean_in_grad = {}
        with model_nnsight.trace() as tracer:
            # Using nnsight's tracer.invoke context, we can batch the clean and the
            # corrupted runs into the same tracing context, allowing us to access
            # information generated within each of these runs within one forward pass

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
            ln_module = model.vit.encoder.layer[layer_idx].layernorm_before
            for head_idx in range(n_heads):
                clean_in_grad[f"a{layer_idx}.h{head_idx}<q>"] = backprop_through_layernorm_functional(
                    clean_in_grad[f"a{layer_idx}.h{head_idx}<q>"], layer_input, ln_module)
                clean_in_grad[f"a{layer_idx}.h{head_idx}<k>"] = backprop_through_layernorm_functional(
                    clean_in_grad[f"a{layer_idx}.h{head_idx}<k>"], layer_input, ln_module)
                clean_in_grad[f"a{layer_idx}.h{head_idx}<v>"] = backprop_through_layernorm_functional(
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

    scores = np.array(scores)
    plt.hist(scores, bins=100)
    plt.show()

    graph.reset(False)
    graph.apply_topn(50, absolute=False, reset=True, prune=False)

    graph.to_image(img_path)

def generate_dataset(data_path, processor, true_label, adversarial_label):

    path = os.path.join(data_path, true_label)
    files = os.listdir(path)
    dpoints_clean = []
    dpoints_adversarial= []
    dpoint_names_saved=set()
    for file in files:
        if (f"Clean-{true_label}" in file) and (f"Adv-{adversarial_label}" in file):
            if "ORIGINAL" in file:
                dpoint_name = os.path.join(path, file.replace("ORIGINAL", "{}"))
            if "ADVERSARIAL" in file:
                dpoint_name = os.path.join(path, file.replace("ADVERSARIAL", "{}"))
            if dpoint_name in dpoint_names_saved:
                continue
            dpoint_names_saved.add(dpoint_name)
            dpoint_clean = Image.open(dpoint_name.format("ORIGINAL")).convert("RGB")
            dpoints_clean.append(dpoint_clean)
            dpoint_adversarial = Image.open(dpoint_name.format("ADVERSARIAL")).convert("RGB")
            dpoints_adversarial.append(dpoint_adversarial)

    input_adversarial = processor(images=dpoints_adversarial, return_tensors="pt")["pixel_values"].to(device)
    input_clean = processor(images=dpoints_clean, return_tensors="pt")["pixel_values"].to(device)

    return input_clean, input_adversarial

def generate_pairwise_dataset(data_path, processor, true_label, distorted_label):
    path_clean = os.path.join(data_path, true_label, true_label)
    path_corrupted = os.path.join(data_path, true_label, distorted_label)
    files_clean = os.listdir(path_clean)
    files_corrupted = os.listdir(path_corrupted)
    dpoints_clean = []
    dpoints_adversarial = []
    for file in files_clean:
        dpoint_clean = Image.open(os.path.join(path_clean, file)).convert("RGB")
        dpoints_clean.append(dpoint_clean)
        dpoint_adversarial = Image.open(os.path.join(path_corrupted, file)).convert("RGB")
        dpoints_adversarial.append(dpoint_adversarial)

    input_adversarial = processor(images=dpoints_adversarial, return_tensors="pt")["pixel_values"].to(device)
    input_clean = processor(images=dpoints_clean, return_tensors="pt")["pixel_values"].to(device)

    return input_clean, input_adversarial

def analyse_pairwise_corruption(model, processor, true_label, distorted_label, true_label_i, distorted_label_i):
    print(f"analyzing {true_label} to  {distorted_label}")
    datas_clean, datas_adversarial = generate_pairwise_dataset(
        "/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset", processor,
        true_label, distorted_label)

    #datas_adversarial, datas_clean = datas_adversarial[:16], datas_clean[:16]

    def metric(logits, unused):
        return (logits[:, distorted_label_i] - logits[:, true_label_i]).mean()

    compGraph = ViTCompGraph.Graph.from_model(model)

    EAP(model, compGraph, datas_clean, datas_adversarial, true_label_i,
        img_path=f"results/{true_label}->{distorted_label}_compgraph.png", metric_fn = metric)

    edge_names = []
    for edge in compGraph.edges:
        edgeObj = compGraph.edges[edge]
        if edgeObj.in_graph:
            edge_names.append(edgeObj.name)

    patched_loss_improvements = {}
    pli = []

    full_loss, full_loss_cache = patch_activations(model, [],
                                         datas_clean, datas_adversarial,
                                         torch.tensor([truth_label for _ in range(len(datas_clean))]),
                                         metric)

    full_clean, full_clean_cache = patch_activations(model, [],
                                                     datas_adversarial, datas_clean,
                                                     torch.tensor([truth_label for _ in range(len(datas_clean))]),
                                                     metric)

    imp = full_loss - full_clean
    stdev = np.std(full_loss_cache - full_clean_cache)
    pli.append({"edge": "full_attack", "improvement": imp, "stdev": stdev, "robustness": imp / stdev})

    for i in range(len(edge_names)):
        edges_to_patch = [edge_names[i]]

        patched_loss, patched_loss_cache = patch_activations(model, edges_to_patch,
                                                         datas_clean, datas_adversarial,
                                                         torch.tensor([truth_label for _ in range(len(datas_clean))]),
                                                         metric)
        imp = full_loss - patched_loss
        stdev = np.std(full_loss_cache - patched_loss_cache)
        if stdev > 0:
            patched_loss_improvements[edge_names[i]] = {"improvement": imp, "stdev": stdev, "robustness": imp/stdev}
            pli.append({"edge": edge_names[i], "improvement": imp, "stdev": stdev, "robustness": imp/stdev})
        else:
            patched_loss_improvements[edge_names[i]] = {"improvement": imp, "stdev": stdev, "robustness": 0}
            pli.append({"edge": edge_names[i], "improvement": imp, "stdev": stdev, "robustness": 0})

    pli = sorted(pli, key=lambda x: x["improvement"], reverse=True)

    json_path = f"results/{true_label}->{distorted_label}_impedgeinfo.json"
    with open(json_path, 'w') as f:
        json.dump(pli, f, indent=4)

    # We'll plot the top 50 edges for clarity
    plot_data = pli
    names = [x["edge"] for x in plot_data]
    robustness_vals = [x["robustness"] for x in plot_data]
    improvement_vals = [x["improvement"] for x in plot_data]

    fig, ax1 = plt.subplots(figsize=(12, 15))

    y_pos = np.arange(len(names))

    # Plot Robustness as the primary metric
    bars = ax1.barh(y_pos, robustness_vals, align='center', color='skyblue', label='Robustness (Imp/Std)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()  # Highest robustness at the top
    ax1.set_xlabel('Robustness Score')
    ax1.set_title(f'Edge Importance: {true_label} -> {distorted_label}')

    ax2 = ax1.twiny()
    ax2.plot(improvement_vals, y_pos, 'ro', markersize=4, label='Raw Improvement')
    ax2.set_xlabel('Raw Loss Improvement')

    fig.tight_layout()
    plt.savefig(f"results/{true_label}->{distorted_label}_impedgeinfo.png")
    plt.close()

def plot_denormalized_img(tensor_img, processor, title="Clean Image"):
    # 1. Clone to avoid modifying the original tensor
    img = tensor_img.clone().detach().cpu()

    # 2. Get mean/std from the processor (handles single float or list)
    # ViT processors usually store these as lists
    mean = torch.tensor(processor.image_mean).view(-1, 1, 1)
    std = torch.tensor(processor.image_std).view(-1, 1, 1)

    # 3. Reverse Normalization: x = (x_norm * std) + mean
    img = img * std + mean

    # 4. Clip to strictly [0, 1] to fix floating point errors
    img = torch.clamp(img, 0, 1)

    # 5. Permute for Matplotlib [C, H, W] -> [H, W, C]
    img = img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id).to(device)
    model.eval()


    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    truth_label = 9
    adversarial_label = 1

    for true_i in range(len(classes)):
        for corr_i in range(len(classes)):
            if true_i == corr_i:
                continue
            if os.path.exists(os.path.join(f"/home/ahmet/PycharmProjects/CMPE492/results/{classes[true_i]}->{classes[corr_i]}_impedgeinfo.png")):
                continue
            analyse_pairwise_corruption(model, processor, classes[true_i], classes[corr_i], true_i, corr_i)

    """data_size = 8
    input_clean, input_adversarial = generate_dataset(
        "/adv_dataset",processor,
        classes[truth_label], classes[adversarial_label])
    input_clean = input_clean[:data_size]
    input_adversarial = input_adversarial[:data_size]
    print(input_clean.shape)

    compGraph = ViTCompGraph.Graph.from_model(model)

    EAP(model, compGraph, input_clean, input_adversarial, truth_label)


    #-----------------TRY PATCHING EDGES-----------------------
    edge_names_score_proxies = []
    for edge in compGraph.edges:
        edgeObj = compGraph.edges[edge]
        if edgeObj.in_graph:
            edge_names_score_proxies.append((edgeObj.score, edgeObj.name))

    edge_names_score_proxies = sorted(edge_names_score_proxies, key=lambda x: x[0], reverse=True)

    patched_losses = []
    for i in range(len(edge_names_score_proxies)):
        edges_to_patch = [edge_names_score_proxies[i][1]]

        patched_loss, patched_logits = patch_activations(model, edges_to_patch,
                                            input_clean, input_adversarial,
                                            torch.tensor([truth_label for _ in range(data_size)]),
                                            F.cross_entropy)
        patched_losses.append(patched_loss)
        print(f"edge patched: {edges_to_patch[0]}, Loss: {patched_loss}")

    edges_to_patch = ["input->a0.h4<v>", "m11->logits"]
    patched_loss, patched_logits = patch_activations(model, edges_to_patch,
                                                     input_clean, input_adversarial,
                                                     torch.tensor([truth_label for _ in range(data_size)]),
                                                     F.cross_entropy)
    print(f"No edges patched: {len(edges_to_patch)}, Loss: {patched_loss}")

    plt.plot(patched_losses)
    plt.show()

    #--------------------TRY MAXIMIZING EDGE EFFECTS-------------------
    plot_denormalized_img(input_clean[0], processor, title="Clean Image")
    # Extract mean/std from processor
    norm_mean = processor.image_mean
    norm_std = processor.image_std

    for edge_name in edges_to_patch:
        los_max = False  # Set to True to see maximization

        # Pass mean/std
        losmaxxed_01 = edge_effect_isolation.maximize_edge_effect_on_metric(
            model, input_clean, edge_name,
            torch.tensor([truth_label for _ in range(data_size)]),
            normalization_mean=norm_mean,
            normalization_std=norm_std,
            increase=los_max
        )

        # --- FIX 3: SIMPLE VISUALIZATION ---
        # The output is already [0, 1], just convert to numpy and plot
        img_np = losmaxxed_01[0].cpu().permute(1, 2, 0).numpy()

        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.title(f"Loss {'Maxxed' if los_max else 'Minned'} on {edge_name}")
        plt.axis('off')
        plt.show()"""




