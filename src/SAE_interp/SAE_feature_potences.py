import json
import os

import numpy as np
import torch
from nnsight import NNsight
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification

from src.SAE.train_sae import SparseAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels

class FeaturePotences:
    @staticmethod
    def calculate_feature_potences(model, layer_SAEs, SAE_activation_stats,  classes, dataloader):
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // n_heads
        layer_head_feature_drags = {}

        def get_class_selection_grad(model_ref, batch):
            data_batch, target_tensor = batch

            grad_vals = None
            with model_ref.trace() as tracer:
                with tracer.invoke(data_batch) as invoker:
                    logits = model.classifier.output
                    grad_vals = model.vit.layernorm.input.grad.save()
                loss = F.cross_entropy(logits, target_tensor)
                loss.backward()
            return grad_vals[:, 0, :]


        for layer_name in layer_SAEs: #range(n_layers):
            if layer_name.startswith("lnb"):
                layer_i = int(layer_name[3:])
            layer_head_feature_drags[f"layer{layer_i}"] = {}
            layer_module = model.vit.encoder.layer[layer_i]
            W_O = layer_module.attention.output.dense.weight
            W_V = layer_module.attention.attention.value.weight
            layer_SAE = layer_SAEs[layer_name]

            # feature vectors decomposed by SAE
            features = torch.cat((layer_SAE.W_dec, layer_SAE.b_dec.unsqueeze(0)), 0)
            feature_mean_acts_nz = torch.tensor(SAE_activation_stats[layer_name]["feature_means_nz"] + [0], device = device)
            feature_mean_diffs = feature_mean_acts_nz

            #this will store W_ov matrices per attention head in layer
            head_matrixes = []
            for head_i in range(n_heads):
                start = head_i * head_dim
                end = (head_i + 1) * head_dim
                W_O_head = W_O[:, start:end]
                W_V_head = W_V[start:end, :]

                #calculate W_ov matrix by this
                OV_matrix_head = W_O_head @ W_V_head
                head_matrixes.append(OV_matrix_head)

                feature_drags = OV_matrix_head @ features.T * feature_mean_diffs
                layer_head_feature_drags[f"layer{layer_i}"][f"head{head_i}"] = feature_drags#.detach().cpu().numpy()

        out_data = {}
        for class_i, clas in enumerate(classes):
            out_data[clas] = {}

            for layer_key, layer_dict in layer_head_feature_drags.items():
                out_data[clas][layer_key] = {}
                for head_key, feature_drags in layer_dict.items():
                    out_data[clas][layer_key][head_key] = (0,0)

            for batch in tqdm(dataloader, f"getting grads for class {clas}"):
                grad_vals = get_class_selection_grad(model, batch)#.detach().cpu().numpy()
                for layer_key, layer_dict in layer_head_feature_drags.items():
                    for head_key, feature_drags in layer_dict.items():
                        effects = (grad_vals @ feature_drags).detach().cpu().numpy()
                        mean_effect = np.mean(effects, axis = 0)
                        sum, n_data = out_data[clas][layer_key][head_key]
                        out_data[clas][layer_key][head_key] = (sum + mean_effect * len(effects), n_data + len(effects))

            for layer_key, layer_dict in layer_head_feature_drags.items():
                for head_key, feature_drags in layer_dict.items():
                    sum, n_data = out_data[clas][layer_key][head_key]
                    out_data[clas][layer_key][head_key] = (sum / n_data).tolist()

        with open(f"/home/ahmet/PycharmProjects/CMPE492/results/feature_potence_calc", "w") as f:
            json.dump(out_data, f, indent=4)

if __name__ == "__main__":
    data_path = '/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset'
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id).to(device)
    model.eval()
    model = NNsight(model)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    batch_size = 32
    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
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


    FeaturePotences.calculate_feature_potences(model, SAEs, SAE_act_stats, classes, dataloader)









