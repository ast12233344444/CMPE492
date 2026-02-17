import json
import math
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification
from nnsight import NNsight
from PIL import Image
from torch.nn import functional as F

import ViTCompGraph
from src.TracingAlgorithms import TracingAlgorithms
from src.experiment_setups import Experiments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO List
#   Activation Patching to see how much the loss decreases Done
#   EAP-IG
#   Edge Corruption Maximization Done
#   Edge Significance (consistency) Done

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


def plot_denormalized_img(tensor_img, processor, title="Clean Image"):
    img = tensor_img.clone().detach().cpu()

    mean = torch.tensor(processor.image_mean).view(-1, 1, 1)
    std = torch.tensor(processor.image_std).view(-1, 1, 1)

    img = img * std + mean

    img = torch.clamp(img, 0, 1)

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
            Experiments.analyse_pairwise_corruption(model, processor, classes[true_i], classes[corr_i], true_i, corr_i)

    """

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




