import math

import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
from nnsight import NNsight
from tqdm import tqdm
from src.TracingAlgorithms import TracingAlgorithms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. DataLoader Setup
def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels

def add_data_points(model, pixel_values, node, head_dim, to_array):
    pixel_values = pixel_values.to(device)

    with model.trace(pixel_values):
        hidden_states = TracingAlgorithms._get_activations(model, node, head_dim)

    activation = hidden_states.detach().cpu().clone().contiguous()

    to_array.append(activation)

def save_all_activations(model, loader, directory, node, type):
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(os.path.join(directory, node)):
        os.makedirs(os.path.join(directory, node))
    if not os.path.exists(os.path.join(directory, node, type)):
        os.makedirs(os.path.join(directory, node, type))

    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    act_buffer = []
    file_counter = 0

    print(f"Starting extraction. Saving to '{output_dir}/' in chunks < 1GB...")

    with torch.no_grad():
        for i, (pixel_values, labels) in enumerate(tqdm(loader, desc="Extracting Activations")):
            add_data_points(model, pixel_values, node, head_dim, act_buffer)

            elsize = math.prod(act_buffer[0].size())

            # If adding this batch exceeds our 900MB limit, save the buffer to disk
            if len(act_buffer) * elsize > 2 ** 28:
                chunk_activations = torch.cat(act_buffer, dim=0)

                save_path = os.path.join(directory, node, type, f"preactivations_chunk_{file_counter:04d}.pt")
                torch.save({
                    "activations": chunk_activations
                }, save_path)

                # Reset buffers
                act_buffer = []
                file_counter += 1

    # Save any remaining tensors in the buffer after the loop finishes
    if act_buffer:
        chunk_activations = torch.cat(act_buffer, dim=0)

        save_path =  os.path.join(directory, node, type, f"preactivations_chunk_{file_counter:04d}.pt")
        torch.save({
            "activations": chunk_activations,
        }, save_path)

    print(f"Extraction complete! Saved {file_counter + (1 if act_buffer else 0)} files.")

if __name__ == "__main__":
    # 1. Directory and Model Setup
    output_dir = "/home/ahmet/PycharmProjects/CMPE492/model_activations"

    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    processor = ViTImageProcessor.from_pretrained(model_id)
    hf_model = ViTForImageClassification.from_pretrained(model_id).to(device)
    hf_model.eval()
    model_nnsight = NNsight(hf_model)

    batch_size = 8
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    loader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    nodes = ["input", "m11"]

    for node in nodes:
        save_all_activations(model_nnsight, loader_train, output_dir, node, "train")
        save_all_activations(model_nnsight, loader_test, output_dir, node, "test")



