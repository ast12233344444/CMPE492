import gc
import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm
from src.TracingAlgorithms import TracingAlgorithms
from src.SAE.train_sae import SparseAutoencoder
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels

def calculate_qk_strengths(model, layer_SAE, SAE_act_stat, feat_pres_stat, layer_i, head_i, include_bias = False):
    """
    Calculates the QK interaction strength matrix for all feature pairs
    in a given attention head, using the full model and layer index.

    Args:
        model: The ViTForImageClassification model (can be wrapped in NNsight).
        layer_SAEs: Dictionary mapping layer names (e.g., "lnb0") to SAE models.
        layer_i: Integer index of the target layer.
        head_i: Integer index of the target attention head.

    Returns:
        qk_strength_matrix: A tensor of shape [n_features + 1, n_features + 1]
                            representing the QK interaction strengths.
    """
    with torch.no_grad():
        layer_module = model.vit.encoder.layer[layer_i]

        head_dim = model.config.hidden_size // model.config.num_attention_heads

        W_Q = layer_module.attention.attention.query.weight
        W_K = layer_module.attention.attention.key.weight

        start = head_i * head_dim
        end = (head_i + 1) * head_dim

        W_Q_head = W_Q[start:end, :]
        W_K_head = W_K[start:end, :]

        if include_bias:
            features = torch.cat((layer_SAE.W_dec, layer_SAE.b_dec.unsqueeze(0)), dim=0)
            feature_acts = torch.tensor(SAE_act_stat["feature_means_nz"] + [1], dtype=torch.float32).unsqueeze(1).to(device)
        else:
            features = layer_SAE.W_dec
            feature_acts = torch.tensor(SAE_act_stat["feature_means_nz"], dtype=torch.float32).unsqueeze(1).to(device)

        features = features * feature_acts

        Q_proj = features @ W_Q_head.t()
        K_proj = features @ W_K_head.t()

        qk_strength_matrix = Q_proj @ K_proj.t()

        feat_pres_matrix = feat_pres_stat.unsqueeze(0) * feat_pres_stat.unsqueeze(1)

    return qk_strength_matrix.detach().cpu().numpy(), feat_pres_matrix.detach().cpu().numpy()

def get_outliers(matrix, bound, ishighbound = True, title = "", path = "hmap.png"):
    graph = {}
    if ishighbound:
        indices = np.argwhere(matrix > bound)
    else:
        indices = np.argwhere(matrix < bound)

    toti = 0
    for indice in indices:
        dest, source = indice
        if dest not in graph:
            graph[dest.item()] = []
        if source not in graph:
            graph[source.item()] = []
        toti += 1
        if ishighbound and matrix[dest, source] < bound:
            raise Exception
        graph[source.item()].append((dest.item(), matrix[dest, source].item()))

    print(toti, len(graph))

    visualize_outlier_heatmap(graph, title = title, path = path)
    return graph


def visualize_outlier_heatmap(graph, title="", path=""):
    nodes = sorted(graph.keys())

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    heatmap_matrix = np.full((len(nodes), len(nodes)), np.nan)

    for source, edges in graph.items():
        col_idx = node_to_idx[source]
        for dest, weight in edges:
            row_idx = node_to_idx[dest]
            heatmap_matrix[row_idx, col_idx] = weight

    plt.figure(figsize=(32, 32))  # Adjust as needed

    ax = sns.heatmap(heatmap_matrix,
                     fmt=".2f",
                     cmap="Reds",
                     linewidths=0.5,
                     linecolor='lightgray',
                     xticklabels=nodes,
                     yticklabels=nodes)

    ax.set_ylabel("Destination (dest) -> Rows")
    ax.set_xlabel("Source (source) -> Columns")
    plt.title(f"Shrunk Outlier Heatmap {title}")
    plt.savefig(path)
    plt.show()

def sample_feature_pairs(qk_values, index_vectors ,n_split, n_feature_per_split):
    assert n_split >= 2
    row_index_vectors, col_index_vectors = index_vectors
    indexes = np.array([row_index_vectors, col_index_vectors]).T

    highest_bound = qk_values[-n_feature_per_split]
    lowest_bound = qk_values[n_feature_per_split]
    bounds = np.linspace(lowest_bound, highest_bound, n_split-1)
    bound_samples = []
    bound_samples.append(indexes[:n_feature_per_split])

    for i in range(1, len(bounds)):
        bound_h = bounds[i]
        bound_h_i = np.searchsorted(qk_values, bound_h)
        bound_l = bounds[i-1]
        bound_l_i = np.searchsorted(qk_values, bound_l)
        slice = indexes[bound_l_i: bound_h_i]
        rng = np.random.default_rng()
        sample_unique = rng.choice(slice, size=min(n_feature_per_split, len(slice)), replace=False)
        bound_samples.append(sample_unique)


    bound_samples.append(indexes[-n_feature_per_split:])

    return bounds, bound_samples

def get_empirical_attention(model, activation_loc, head_i, SAE, dataloader, feature_pairs):
    layer_i = int(activation_loc[3:])

    model.config.output_attentions = True
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    pairs = [tuple(pair) for pair in feature_pairs]
    attention_sums = {pair: 0.0 for pair in pairs}
    attention_counts = {pair: 0 for pair in pairs}

    # Wrap in NNsight if it isn't already
    if not hasattr(model, 'trace'):
        from nnsight import NNsight
        model = NNsight(model)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Getting Empirical Attention (Layer {layer_i}, Head {head_i})"):
            x_batch, _ = batch
            x_batch = x_batch.to(model.device)

            with model.trace() as tracer:
                with tracer.invoke(x_batch) as invoker:
                    activations = TracingAlgorithms._get_activations(model, activation_loc, head_dim)

                    encoded_features, _ = SAE(activations)
                    encoded_features = encoded_features.save()

                    attention_probs = model.vit.encoder.layer[layer_i].attention.attention.output[1].save()

            feats = encoded_features.value

            attns = attention_probs.value[:, head_i, :, :]

            feats_present = (feats > 0)

            for f_q, f_k in pairs:
                q_present = feats_present[:, :, f_q]
                k_present = feats_present[:, :, f_k]

                pair_mask = q_present.unsqueeze(2) & k_present.unsqueeze(1)

                attention_sums[(f_q, f_k)] += (attns * pair_mask).sum().item()
                attention_counts[(f_q, f_k)] += pair_mask.sum().item()

    empirical_attentions = {}
    for pair in pairs:
        if attention_counts[pair] > 0:
            empirical_attentions[pair] = attention_sums[pair] / attention_counts[pair]
        else:
            empirical_attentions[pair] = 0.0

    return empirical_attentions

def run_stats_big_matrix(matrix, L0_norm, layer, head, title, out_dir, s):
    flat_matrix = matrix.flatten()
    lbound = np.mean(flat_matrix) - s * np.sqrt(L0_norm) * np.std(flat_matrix)
    hbound = np.mean(flat_matrix) + s * np.sqrt(L0_norm) * np.std(flat_matrix)

    get_outliers(matrix, hbound, title=f"layer {layer} head {head} {title} heatmap",
                 path=f"{out_dir}/l{layer}_h{head}/{title}_heatmap.png")

    plt.figure(figsize=(10, 6))
    plt.hist(flat_matrix[(flat_matrix > lbound) & (flat_matrix < hbound)], bins=150, log=True, color='skyblue', alpha=0.7)
    plt.hist(flat_matrix[(flat_matrix < lbound) | (flat_matrix > hbound)], bins=150, log=True, color='red', alpha=0.7)

    plt.title(f"Distribution of QK {title} (Layer {layer}, Head {head})", fontsize=14)
    plt.xlabel("Interaction Strength", fontsize=12)
    plt.ylabel("Frequency (Log Scale)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/l{layer}_h{head}/{title}_histogram.png", dpi=300)
    plt.show()


def get_th_emp_plots(layer, head, model, SAEs, SAE_act_stats, feat_presence_stats, dataloader, out_dir, empirical = False, cumulative= True):
    qk_strength_matrix, feat_presence_matrix = calculate_qk_strengths(model, SAEs[f"lnb{layer}"], SAE_act_stats[f"lnb{layer}"],
                                                                      feat_presence_stats, layer, head)
    feat_presence_matrix = np.log(np.maximum(feat_presence_matrix, 1e-9))
    os.makedirs(f"{out_dir}/l{layer}_h{head}/", exist_ok=True)
    L0_norm = SAE_act_stats[f"lnb{layer}"]["l0_mean"]

    if cumulative:
        qk_strength_matrix = qk_strength_matrix * np.exp(feat_presence_matrix)

    qk_values = qk_strength_matrix.flatten()
    #flat_presence_values = feat_presence_matrix.flatten()

    run_stats_big_matrix(qk_strength_matrix, L0_norm, layer, head, "interaction_strength", out_dir, 5)
    #run_stats_big_matrix(qk_strength_matrix * np.exp(feat_presence_matrix), L0_norm, layer, head, "cumulative_strength", out_dir, 5)

    sorted_1d_indices = np.argsort(qk_values)
    qk_values = qk_values[sorted_1d_indices]
    #flat_presence_values = flat_presence_values[sorted_1d_indices]

    row_indices, col_indices = np.unravel_index(sorted_1d_indices, qk_strength_matrix.shape)

    bounds, bound_samples = sample_feature_pairs(qk_values, (row_indices, col_indices), 10, 100)
    with open(f"{out_dir}/l{layer}_h{head}/bound_samples.json", "w") as f:
        json.dump([bound_sample.tolist() for bound_sample in bound_samples], f, indent=4)

    if empirical:

        bound_samples_flat = np.concatenate(bound_samples)
        empirical_attetnion_data = get_empirical_attention(model, f"lnb{layer}", head, SAEs[f"lnb{layer}"], dataloader,
                                                           bound_samples_flat)
        bound_groups = [[] for _ in range(len(bound_samples))]

        for i in range(len(bound_samples)):
            for feature_pair in empirical_attetnion_data:
                if i > 0:
                    if qk_strength_matrix[feature_pair[0]][feature_pair[1]] < bounds[i - 1]:
                        continue
                if i < len(bound_samples) - 1:
                    if qk_strength_matrix[feature_pair[0]][feature_pair[1]] > bounds[i]:
                        continue

                bound_groups[i].append(empirical_attetnion_data[feature_pair])
        del row_indices, col_indices, sorted_1d_indices


    if empirical:
        # Filter out empty groups so Matplotlib doesn't throw an error
        valid_groups = [group for group in bound_groups if len(group) > 0]

        plt.figure(figsize=(10, 6))

        # Standard sequential boxplot (1, 2, 3...)
        plt.boxplot(valid_groups, showfliers=False, patch_artist=True)

        plt.title(f"Empirical Attention by QK Strength Bin (Layer {layer}, Head {head})")
        plt.xlabel("QK Strength Bins (Lowest to Highest)")
        plt.ylabel("Empirical Attention")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/l{layer}_h{head}/emprirical_result.png", dpi=300)
        plt.show()

    del qk_strength_matrix, feat_presence_matrix, qk_values
    gc.collect()


if __name__ == "__main__":
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager").to(device)
    processor = ViTImageProcessor.from_pretrained(model_id)
    model.eval()
    batch_size = 64
    dataset = datasets.CIFAR10(root='../data', train=True, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    SAE_dir = "/home/ahmet/PycharmProjects/CMPE492/saved_models"
    SAE_act_stats_dir = f"/home/ahmet/PycharmProjects/CMPE492/model_activation_stats"
    results_dir = f"/home/ahmet/PycharmProjects/CMPE492/results/QK_circuit_analysis"
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

    with open("/home/ahmet/PycharmProjects/CMPE492/results/avg_attention_scores.json", "r") as f:
        feat_presence_stats = json.load(f)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    layerwise_presence_stats = {}
    for layer in range(12):
        tot_probs = None
        for cls in classes:
            if tot_probs is None:
                tot_probs = torch.tensor(feat_presence_stats[f"class_presence_probs"][cls][f"layer{layer}"][f"head0"], device=device)
            else:
                tot_probs += torch.tensor(feat_presence_stats[f"class_presence_probs"][cls][f"layer{layer}"][f"head0"], device=device)
        tot_probs /= len(classes)
        layerwise_presence_stats[layer] = tot_probs

    for head in tqdm(range(12), "heads"):
        for layer in tqdm(range(0, 12, 4), "layers"):
            get_th_emp_plots(layer, head, model, SAEs, SAE_act_stats, layerwise_presence_stats[layer], dataloader, results_dir, empirical = True, cumulative = False)














