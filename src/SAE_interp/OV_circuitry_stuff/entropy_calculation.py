import os
import json
from matplotlib import pyplot as plt
from nnsight import NNsight
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import ViTImageProcessor, ViTForImageClassification

from src.SAE.train_sae import SparseAutoencoder
from src.TracingAlgorithms import TracingAlgorithms
import math
import numpy as np
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels


def get_feature_scattering_coefficient(grid):
    # Ensure grid is a NumPy array and reshape to 14x14
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    else:
        grid = np.array(grid)

    grid_2d = grid.reshape((14, 14))

    mask = grid_2d > 0
    total_nonzero = np.sum(mask)

    if total_nonzero == 0:
        return 0.0

    visited = np.zeros_like(mask, dtype=bool)
    num_blobs = 0

    for r in range(14):
        for c in range(14):
            if mask[r, c] and not visited[r, c]:
                num_blobs += 1
                stack = [(r, c)]

                while stack:
                    curr_r, curr_c = stack.pop()

                    if 0 <= curr_r < 14 and 0 <= curr_c < 14:
                        if mask[curr_r, curr_c] and not visited[curr_r, curr_c]:
                            visited[curr_r, curr_c] = True

                            stack.extend([
                                (curr_r - 1, curr_c),  # Up
                                (curr_r + 1, curr_c),  # Down
                                (curr_r, curr_c - 1),  # Left
                                (curr_r, curr_c + 1)  # Right
                            ])

    return float(num_blobs / total_nonzero)

def get_under_class_entropies(model, dataloader, SAEs, target_features,
                              n_classes=10, p_vals = [0, 1, 2, 3, 4, 5],
                              out_dir="/home/ahmet/PycharmProjects/CMPE492/results/OV_feature_entropies", vis = True):
    os.makedirs(out_dir, exist_ok=True)
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    feature_activations = {}
    feature_scattering = {}

    for layer, features in target_features.items():
        feature_activations[layer] = {}
        feature_scattering[layer] = {}

        feature_list = features.tolist() if isinstance(features, torch.Tensor) else list(features)
        for feature in feature_list:
            feature_activations[layer][feature] = {}
            feature_scattering[layer][feature] = []
            for p in p_vals:
                feature_activations[layer][feature][p] = {cls: 0 for cls in range(n_classes)}

    with torch.no_grad():
        for batch in tqdm(dataloader, "Processing batches..."):
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(model.device), y_batch.to(model.device)

            saved_encoded_features = {}

            with model.trace() as tracer:
                with tracer.invoke(x_batch):
                    for layer in target_features:
                        l_i = int(layer[1:])
                        activations = TracingAlgorithms._get_activations(model, f"lnb{l_i}", head_dim)

                        f_acts, _ = SAEs[f"lnb{l_i}"](activations)
                        saved_encoded_features[layer] = f_acts.save()

            # The trace context has exited; proxies now hold actual tensor values
            for layer, features in target_features.items():
                acts = saved_encoded_features[layer].value
                acts = acts.detach().cpu().numpy()
                feature_list = features.tolist() if isinstance(features, torch.Tensor) else list(features)

                for i in range(len(y_batch)):
                    for feature in feature_list:
                        grid_flattened = acts[i, 1:, feature]
                        scattering_val = get_feature_scattering_coefficient(grid_flattened)
                        if scattering_val > 0:
                            feature_scattering[layer][feature].append(np.log(scattering_val))

                if acts.ndim == 3:
                    acts = acts.sum(axis=1)


                for i, cls in enumerate(y_batch.tolist()):
                    for feature in feature_list:
                        val = acts[i, feature].item()
                        for p in p_vals:
                            if val > 0:
                                feature_activations[layer][feature][p][cls] += val**p

    feature_entropies = {}
    for layer in feature_activations:
        feature_entropies[layer] = {}
        for feature, class_dists in feature_activations[layer].items():
            for p, class_dist in class_dists.items():
                if p not in feature_entropies[layer]:
                    feature_entropies[layer][p] = {}
                total_activation = sum(class_dist.values())
                entropy = 0.0

                if total_activation > 0:
                    for cls, count in class_dist.items():
                        prob = count / total_activation
                        if prob > 0:
                            entropy -= prob * math.log2(prob)

                feature_entropies[layer][p][feature] = entropy

    scat_vals = {}
    for layer in feature_scattering:
        scat_vals[layer] = {}
        for feature in feature_scattering[layer]:
            if len(feature_scattering[layer][feature]) > 0:
                scat_vals[layer][feature] = np.mean(feature_scattering[layer][feature])

    if vis:
        for p in p_vals:
            plt.figure(figsize=(10, 6))
            for layer, f_entropies in feature_entropies.items():

                # Initialize synchronized lists for plotting
                entropy_values = []
                scattering_values = []

                for feature, entropy in f_entropies[p].items():
                    scat_list = feature_scattering[layer][feature]

                    # Only plot if we actually recorded scattering values for this feature
                    if len(scat_list) > 0:
                        entropy_values.append(entropy)
                        # Average the scattering values across all images for this feature
                        scattering_values.append(float(np.mean(scat_list)))

                plt.scatter(entropy_values, scattering_values, label=layer)

            plt.title(f"Distribution of Feature Class-Entropies by Layer (p = {p})")
            plt.xlabel("Shannon Entropy (bits)")
            plt.ylabel("Mean Log Scattering Coefficient")
            plt.legend(title="Layer")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/feature_entropies_p{p}.png")
            plt.show()

    print(scat_vals)
    return feature_entropies, scat_vals

if __name__ == "__main__":
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager").to(device)
    model.eval()
    model = NNsight(model)

    cutoff_effect = 1e-4
    n_toks = 197
    n_heads = 12
    batch_size = 32
    feature_potence_path = "/home/ahmet/PycharmProjects/CMPE492/results/feature_potence_calc.json"
    average_attention_data_path = "/home/ahmet/PycharmProjects/CMPE492/results/avg_attention_scores.json"

    feature_potence_data = json.load(open(feature_potence_path))
    average_attention_data = json.load(open(average_attention_data_path))
    feature_presence_data = average_attention_data["class_presence_probs"]
    average_attention_data = average_attention_data["class_avg_attentions"]


    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    features_per_layer = {}
    for layer_i in [8, 9, 10, 11]:
        full_layer_effect_datas_by_class = []
        features_per_layer[f"l{layer_i}"] = set()

        for head_i in range(n_heads):
            att_data = None
            for i in range(len(classes)):
                if att_data is None:
                    div_coeff = np.zeros(len(average_attention_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"]) + 1)
                    att_data = np.zeros(len(average_attention_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"]) + 1)
                pres = np.array(feature_presence_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"] + [1])
                att_data += pres * np.array(average_attention_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"] + [1 / n_toks])
                div_coeff += pres
            att_data /= (div_coeff + 1e-9)

            for i in range(len(classes)):
                pot_data = np.array(feature_potence_data[classes[i]][f"layer{layer_i}"][f"head{head_i}"])
                if head_i == 0:
                    full_layer_effect_datas_by_class.append(att_data * pot_data)
                else:
                    full_layer_effect_datas_by_class[i] += att_data * pot_data

        for i in range(len(classes)):
            for j in range(len(full_layer_effect_datas_by_class[i])):
                if abs(full_layer_effect_datas_by_class[i][j]) > cutoff_effect:
                    features_per_layer[f"l{layer_i}"].add(j)
        features_per_layer[f"l{layer_i}"] = torch.tensor(list(features_per_layer[f"l{layer_i}"]), device=model.device)

    dataset = datasets.CIFAR10(root='../data', train=True, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    SAE_dir = "/home/ahmet/PycharmProjects/CMPE492/saved_models"
    SAE_act_stats_dir = f"/home/ahmet/PycharmProjects/CMPE492/model_activation_stats"
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
            SAE_model = SparseAutoencoder(expansion_factor=SAE_metadata["expansion_factor"])
            SAE_model.load_state_dict(SAE_metadata["model_state_dict"])
            SAE_model = SAE_model.to(device)
            SAEs[layer_name] = SAE_model

    get_under_class_entropies(model, dataloader, SAEs, features_per_layer, len(classes))




