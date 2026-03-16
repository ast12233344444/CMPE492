import heapq
import json

import matplotlib

from src.SAE_interp.FeatureVis.FVisPlottingFuncs import save_top_k_visualizations, save_qk_pair_visualizations

matplotlib.use('Agg')
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification

from src.SAE.train_sae import SparseAutoencoder
from src.TracingAlgorithms import TracingAlgorithms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels

def manage_heap(heap, vals, feature_acts, tie_breaker, x_batch, n_samples):
    if len(heap) == n_samples:
        feature_min_acceptable = heap[0][0]
    else:
        feature_min_acceptable = 0
    feature_indices = torch.nonzero((vals > feature_min_acceptable).float())
    for feature_indice in feature_indices:
        if vals[feature_indice] < feature_min_acceptable:
            continue
        heapq.heappush(heap, (
            vals[feature_indice].item(),
            tie_breaker,
            x_batch[feature_indice],
            feature_acts[feature_indice]))
        tie_breaker += 1
        if len(heap) > n_samples:
            heapq.heappop(heap)
        if len(heap) == n_samples:
            feature_min_acceptable = heap[0][0]

def get_feature_maximisers(model, dataloader, SAEs, locations, n_samples):
    layers_to_record = set()
    for location in locations:
        layer_name = location.split("-")[0]
        if layer_name.startswith("lnb"):
            layers_to_record.add(int(layer_name[3:]))
        else:
            raise Exception
    location_maximizers_by_max = {layer: {} for layer in layers_to_record}
    location_maximizers_by_avg = {layer: {} for layer in layers_to_record}
    for location in locations:
        layer_name = location.split("-")[0]
        layer =int(layer_name[3:])
        location_maximizers_by_max[layer][int(location.split("-")[1])] = []
        location_maximizers_by_avg[layer][int(location.split("-")[1])] = []


    layers_to_record = sorted(list(layers_to_record))
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    if not hasattr(model, 'trace'):
        from nnsight import NNsight
        model = NNsight(model)

    tie_breaker = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, "processing batches..."):
            x_batch, _ = batch
            x_batch = x_batch.to(model.device)
            encoded_features = {}

            with model.trace() as tracer:
                with tracer.invoke(x_batch) as invoker:
                    for layer in layers_to_record:
                        activations = TracingAlgorithms._get_activations(model, f"lnb{layer}", head_dim)

                        feats, _ = SAEs[f"lnb{layer}"](activations)
                        encoded_features[layer] = feats.save()

            for layer, encoding in encoded_features.items():
                encoding = encoding.value
                for feature_no in location_maximizers_by_max[layer]:
                    feature_activations = encoding[:, :, feature_no]
                    feature_max_activations = torch.max(feature_activations, dim = 1)[0]
                    feature_avg_activations = torch.mean(feature_activations, dim = 1)

                    manage_heap(location_maximizers_by_max[layer][feature_no], feature_max_activations,
                                feature_activations, tie_breaker, x_batch, n_samples)

                    manage_heap(location_maximizers_by_avg[layer][feature_no], feature_avg_activations,
                                feature_activations, tie_breaker, x_batch, n_samples)

                    tie_breaker += x_batch.shape[0]
    return location_maximizers_by_max, location_maximizers_by_avg



def get_features_from_csv(files):
    features = set()
    for file in files:
        layer = int(file.split(".")[0].split("_")[-1])
        table = pd.read_csv(file)
        for i in range(len(table)):
            no_feature = int(table.iloc[i]["Feature"].split("_")[1])
            features.add(f"lnb{layer}-{no_feature}")
    return list(features)


def OV_circuit_plotting_routine(model, dataloader, SAEs):
    files = ["/home/ahmet/PycharmProjects/CMPE492/results/OV_dump/feature_potence_8.csv",
             "/home/ahmet/PycharmProjects/CMPE492/results/OV_dump/feature_potence_9.csv"]
    locations = get_features_from_csv(files)
    print(locations)

    location_maximisers_max, location_maximisers_avg = get_feature_maximisers(model, dataloader, SAEs, locations, 5)
    save_top_k_visualizations(location_maximisers_max, location_maximisers_avg)

def QK_circuit_plotting_routine(model, dataloader, SAEs, nodes = None):
    n_layers = model.config.num_hidden_layers
    QK_res_dir = "/results/QK_circuit_analysis"
    if nodes == None:
        nodes = {layer: [] for layer in range(n_layers)}
        dir_QK_res = os.listdir(QK_res_dir)
        for dir_QK in dir_QK_res:
            layer, head = dir_QK.split("_")
            nodes[int(layer[1:])].append(int(head[1:]))

    for layer, node_heads in nodes.items():
        if len(node_heads) == 0:
            continue

        features = set()
        feature_pairs_by_head = {}
        for head in node_heads:
            feature_pairs = []
            json_path = os.path.join(QK_res_dir, f"l{layer}_h{head}", "bound_samples.json")

            with open(json_path, 'r') as f:
                json_data = json.load(f)

            feature_pairs_cared = json_data[-1]
            for pair in feature_pairs_cared:
                features.add(pair[0])
                features.add(pair[1])
                feature_pairs.append(pair)
            feature_pairs_by_head[head] = feature_pairs
        features = list(features)
        locations = []
        for feature in features:
            locations.append(f"lnb{layer}-{feature}")

        location_maximisers_max, location_maximisers_avg = get_feature_maximisers(model, dataloader, SAEs, locations, 3)

        for head in node_heads:
            head_feature_pairs = feature_pairs_by_head[head]
            save_dir = os.path.join(QK_res_dir, f"l{layer}_h{head}", "attracting_pair_maximizers")

            save_qk_pair_visualizations(
                layer=layer,
                head=head,
                feature_pairs=head_feature_pairs,
                location_maximizers_max=location_maximisers_max,
                location_maximisers_avg=location_maximisers_avg,
                save_dir=save_dir
            )


if __name__ == "__main__":
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    SAE_dir = "/saved_models"
    saved_SAE_dir = os.listdir(SAE_dir)
    model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager").to(device)
    processor = ViTImageProcessor.from_pretrained(model_id)
    model.eval()
    batch_size =32
    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)



    #locations = ["lnb11-10678", "lnb11-709", "lnb11-3544","lnb11-94", "lnb11-8724", "lnb11-3059", "lnb11-612",
    #             "lnb10-3620", "lnb10-4500", "lnb10-3061", "lnb10-1759", "lnb10-2385", "lnb10-8176", "lnb10-9032"]

    SAEs = {}
    for saved_SAE_name in saved_SAE_dir:
        if "lnb" in saved_SAE_name:
            layer_name = saved_SAE_name.split("_")[1]
            SAE_metadata = torch.load(os.path.join(SAE_dir, saved_SAE_name))
            SAE_model = SparseAutoencoder(expansion_factor=SAE_metadata["expansion_factor"])
            SAE_model.load_state_dict(SAE_metadata["model_state_dict"])
            SAE_model = SAE_model.to(device)
            SAEs[layer_name] = SAE_model

    #OV_circuit_plotting_routine(model, dataloader, SAEs)
    QK_circuit_plotting_routine(model, dataloader, SAEs)

