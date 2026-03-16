import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification
from nnsight import NNsight
import torch.nn.functional as F

from src.SAE.train_sae import SparseAutoencoder
from src.SAE_interp.OV_circuitry_stuff.entropy_calculation import get_under_class_entropies
from src.TracingAlgorithms import TracingAlgorithms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels

class FeatureNullifiedVit(nn.Module):
    def __init__(self, model, SAEs, features_to_nullify):
        super().__init__()
        self.model = model
        self.SAEs = SAEs
        self.features_to_nullify = features_to_nullify
        n_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // n_heads

    def forward(self, x):
        layers = []
        for layer in self.features_to_nullify:
            layers.append(layer)
        layers = sorted(layers)
        with self.model.trace() as tracer:
            with tracer.invoke(x):
                for layer in layers:
                    activations = TracingAlgorithms._get_activations(self.model, f"lnb{layer}", self.head_dim)

                    f_acts, _ = self.SAEs[f"lnb{layer}"](activations)
                    subt_vector = f_acts[:, :, self.features_to_nullify[layer]] @ self.SAEs[f"lnb{layer}"].W_dec[self.features_to_nullify[layer], :]

                    TracingAlgorithms._set_activations(self.model, f"lnb{layer}", self.head_dim, activations - subt_vector)
                outputs = self.model.classifier.output.save()
        return outputs.value


def evaluate_dataset(model, dataloader, is_nullified_wrapper=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_nullified_wrapper:
                logits = model(images)
            else:
                outputs = model(images)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def compare_model_performance(base_model, train_loader, test_loader):
    """
    Compares the Base ViT and the FeatureNullifiedVit on Train and Test sets.
    """
    #print("\nStarting evaluation of Feature-Nullified Model...")
    #null_train_loss, null_train_acc = evaluate_dataset(nullified_model, train_loader, is_nullified_wrapper=True)
    #null_test_loss, null_test_acc = evaluate_dataset(nullified_model, test_loader, is_nullified_wrapper=True)

    print("\nStarting evaluation of Base Model...")
    base_train_loss, base_train_acc = evaluate_dataset(base_model, train_loader, is_nullified_wrapper=False)
    base_test_loss, base_test_acc = evaluate_dataset(base_model, test_loader, is_nullified_wrapper=False)


    print("\n" + "=" * 60)
    print(f"{'Metric':<20} | {'Base Model':<15} | {'Nullified Model':<15}")
    print("-" * 60)
    print(f"{'Train Accuracy':<20} | {base_train_acc:>14.4f}% |")
    print(f"{'Train Loss (CE)':<20} | {base_train_loss:>14.4f}  |")
    print("-" * 60)
    print(f"{'Test Accuracy':<20} | {base_test_acc:>14.4f}% |")
    print(f"{'Test Loss (CE)':<20} | {base_test_loss:>14.4f}  |")
    print("=" * 60)

    return {"base": {"train_acc": base_train_acc, "train_loss": base_train_loss, "test_acc": base_test_acc,
                 "test_loss": base_test_loss} }

def get_features_from_entropies(entropies, scatterings, p = 3, thresh = 0.25, thresh_scat = -0.6):
    layer_features = {}
    for layer in entropies:
        layer_i = int(layer[1:])
        layer_features[layer_i] = set()
        for feature, feat_ent in entropies[layer][p].items():
            if feat_ent > thresh:
                layer_features[layer_i].add(feature)
        for feature, scat in scatterings[layer].items():
            if scat > thresh_scat:
                #print(f"scat: {scat}, feature: {feature} adding...")
                layer_features[layer_i].add(feature)
    for layer_i in layer_features:
        layer_features[layer_i] = torch.tensor(list(layer_features[layer_i]), dtype=torch.int32, device=device)
    return layer_features


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
    for layer_i in [11]:
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

    dataset_train = datasets.CIFAR10(root='../data', train=True, download=True)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    dataset_test = datasets.CIFAR10(root='../data', train=False, download=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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

    feature_entropy_data, feature_scattering_data = get_under_class_entropies(model, dataloader_train, SAEs, features_per_layer, len(classes))

    compare_model_performance(
        base_model=model,
        train_loader=dataloader_train,
        test_loader=dataloader_test,
    )
    for p in [3,4,5]:
        for thresh in [0.05, 0.1, 0.25, 0.5, 1]:
            print(f"testing for p={p} thresh={thresh}")
            nullify_features_data = get_features_from_entropies(feature_entropy_data, feature_scattering_data, p = p, thresh = thresh)

            wrapped_model = FeatureNullifiedVit(model, SAEs, nullify_features_data)

            compare_model_performance(base_model=wrapped_model,
                                      train_loader=dataloader_train,
                                      test_loader=dataloader_test)

