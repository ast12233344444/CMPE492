import json
import shutil

import numpy as np
from nnsight import NNsight
from torchvision import datasets
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTImageProcessor, ViTForImageClassification

from src.SAE.save_activations import save_all_activations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import torch
import random
from torch.utils.data import IterableDataset, DataLoader

def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels

class BufferedActivationDataset(IterableDataset):
    def __init__(self, activation_dir, buffer_size_chunks=1):
        """
        Args:
            activation_dir: Path to your .pt files.
            buffer_size_chunks: How many 1GB chunks to hold in RAM at once.
        """
        self.files = sorted([
            os.path.join(activation_dir, f)
            for f in os.listdir(activation_dir)
            if f.endswith('.pt')
        ])
        # Randomize file order for better global shuffling
        random.shuffle(self.files)

    def __iter__(self):
        for file_path in self.files:
            # Load the ~1GB chunk once
            data = torch.load(file_path, map_location="cpu")['activations']

            # Reshape from [N, 197, 768] to [N*197, 768]
            flat_data = data.view(-1, data.size(-1))

            # Local shuffle of tokens within this chunk
            indices = torch.randperm(flat_data.size(0))

            for idx in indices:
                yield flat_data[idx]



class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=768, expansion_factor=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor

        # 1. Decoder Bias (b_dec): Used to center the data before encoding
        # and added back after decoding.
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        # 2. Encoder: Linear(W_enc) + b_enc
        self.encoder = nn.Linear(input_dim, self.hidden_dim)

        # 3. Decoder: Weight matrix (W_dec) only.
        # We handle the multiplication manually to ensure we can easily
        # normalize columns to unit norm.
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(self.hidden_dim, input_dim)
        ))

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        x_centered = x - self.b_dec

        encoded = F.relu(self.encoder(x_centered))

        decoded = F.linear(encoded, self.W_dec.t()) + self.b_dec

        return encoded, decoded

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """
        Standard SAE training practice: Constrain decoder columns to unit norm.
        This prevents the model from minimizing L1 loss by simply scaling up
        W_dec and scaling down feature activations (z).
        """
        # Normalize weights
        norms = torch.norm(self.W_dec, dim=1, keepdim=True)
        self.W_dec.div_(norms)

        # Also remove the projection of the gradient onto the weight vector
        # to keep updates stable on the unit hypersphere.
        if self.W_dec.grad is not None:
            grad_proj = (self.W_dec.grad * self.W_dec).sum(dim=1, keepdim=True) * self.W_dec
            self.W_dec.grad.sub_(grad_proj)


def evaluate_sae(model, dataloader, l1_coeff):
    model.eval()
    total_mse = 0
    total_l1 = 0
    total_l0 = 0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            encoded, decoded = model(x)

            mse_loss = F.mse_loss(decoded, x)
            l1_loss = encoded.abs().sum(dim=-1).mean()
            l0 = (encoded > 0).float().sum(dim=-1).mean()

            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
            total_l0 += l0.item()
            batch_count += 1

            # Limit evaluation steps if test set is massive
            if batch_count >= 100:
                break

    return total_mse / batch_count, total_l1 / batch_count, total_l0 / batch_count

def train_sae(model, train_loader, test_loader, l1_coeff, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        model.train()
        total_mse, total_l1, batch_count = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            x = batch.to(device)  # Fixed: Removed [0]

            encoded, decoded = model(x)
            mse_loss = F.mse_loss(decoded, x)
            l1_loss = encoded.abs().sum(dim=-1).mean()
            loss = mse_loss + l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()
            model.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()

            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
            batch_count += 1

            if batch_count % 50 == 0:
                pbar.set_postfix({"mse": f"{mse_loss.item():.4f}", "l1": f"{l1_loss.item():.2f}"})

        # Run Evaluation
        test_mse, test_l1, test_l0 = evaluate_sae(model, test_loader, l1_coeff)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train -> MSE: {total_mse / batch_count:.6f} | L1: {total_l1 / batch_count:.4f}")
        print(f"  Test  -> MSE: {test_mse:.6f} | L1: {test_l1:.4f} | L0: {test_l0:.1f}")

def extract_feature_stats(SAE, dataloader, out_path, max_samples=100_000):
    # 1. Pre-allocate memory using a flat numpy array (float32 saves 50% RAM)
    nonzero_acts = np.zeros((SAE.hidden_dim, max_samples), dtype=np.float32)
    nz_counts = np.zeros(SAE.hidden_dim, dtype=np.int32)

    resid_norms = []
    l0_norms = []

    with torch.no_grad():
        for batch in tqdm(dataloader, "Extracting stats..."):
            batch = batch.to(device)
            encoded, decoded = SAE(batch)

            batch_l0 = (encoded > 0).float().sum(dim=1).detach().cpu().numpy()

            batch_resid_norms = torch.norm(batch - decoded, p=2, dim=1).detach().cpu().numpy()

            if len(resid_norms) < max_samples:
                resid_norms.append(batch_resid_norms)
                l0_norms.append(batch_l0)

            encoded_cpu = encoded.detach().cpu().numpy()

            for hidden_idx in range(SAE.hidden_dim):
                if nz_counts[hidden_idx] >= max_samples:
                    continue

                slice_vals = encoded_cpu[:, hidden_idx]
                nz_vals = slice_vals[slice_vals > 0]
                n_nz = len(nz_vals)

                if n_nz > 0:
                    space_left = max_samples - nz_counts[hidden_idx]
                    to_add = min(n_nz, space_left)

                    start_idx = nz_counts[hidden_idx]
                    end_idx = start_idx + to_add

                    # 2. Insert directly into pre-allocated array
                    nonzero_acts[hidden_idx, start_idx:end_idx] = nz_vals[:to_add]
                    nz_counts[hidden_idx] += to_add
    resid_norms = np.concatenate(resid_norms, axis = 0)
    l0_norms = np.concatenate(l0_norms, axis = 0)

    feature_means_nz, feature_99_nz, feature_999_nz, feature_maxs = [], [], [], []

    for hidden_idx in range(SAE.hidden_dim):
        acts = np.array(nonzero_acts[hidden_idx][:nz_counts[hidden_idx]])
        # Existing Non-Zero Stats
        if len(acts) > 0:
            feature_means_nz.append(float(np.mean(acts)))
            feature_99_nz.append(float(np.quantile(acts, 0.99)))
            feature_999_nz.append(float(np.quantile(acts, 0.999)))
            feature_maxs.append(float(np.max(acts)))
        else:
            feature_means_nz.append(0.0)
            feature_99_nz.append(0.0)
            feature_999_nz.append(0.0)
            feature_maxs.append(0.0)

    # Convert global metrics to arrays for easy stat calculation
    resid_arr = np.array(resid_norms)
    l0_arr = np.array(l0_norms)

    out_dict = {
        "feature_means_nz": feature_means_nz,
        "feature_99_nz": feature_99_nz,
        "feature_999_nz": feature_999_nz,
        "feature_maxs": feature_maxs,

        # Residual Stats
        "resid_l2_mean": float(np.mean(resid_arr)),
        "resid_l2_99": float(np.quantile(resid_arr, 0.99)),
        "resid_l2_999": float(np.quantile(resid_arr, 0.999)),
        "resid_l2_max": float(np.max(resid_arr)),

        # L0 Norm Stats
        "l0_mean": float(np.mean(l0_arr)),
        "l0_99": float(np.quantile(l0_arr, 0.99)),
        "l0_999": float(np.quantile(l0_arr, 0.999)),
        "l0_max": float(np.max(l0_arr)),
    }

    with open(out_path, "w") as f:
        json.dump(out_dict, f)

if __name__ == "__main__":
    project_dir = os.getcwd()
    # Create a directory for saved models if it doesn't exist
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    processor = ViTImageProcessor.from_pretrained(model_id)
    hf_model = ViTForImageClassification.from_pretrained(model_id).to(device)
    hf_model.eval()
    model = NNsight(hf_model)

    batch_size = 8
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    loader_test_img = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    loader_train_img = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    save_dir = f"{project_dir}/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    train = True
    export_act = True

    # 1. Hyperparameters
    n_attention_layers = model.config.num_hidden_layers
    input_dim = 768  # ViT-Base hidd                                                                        en dim
    expansion_factors = [8]
    l1_coefficient = 1e-4  # Adjust lambda based on target sparsity (L0)
    batch_size = 4096  # SAEs benefit from large batches
    learning_rate = 3e-4
    nodes = ["lnb1"]#f"lnb{i}" for i in range(n_attention_layers)]

    if train:
        for node in nodes:
            base_path = f"{project_dir}/model_activations"
            save_all_activations(model, loader_train_img, base_path, node, "train")
            save_all_activations(model, loader_test_img, base_path, node, "test")
            for expansion_factor in expansion_factors:
                SAE_model = SparseAutoencoder(input_dim, expansion_factor).to(device)
                optimizer = optim.Adam(SAE_model.parameters(), lr=learning_rate)


                train_loader = DataLoader(BufferedActivationDataset(f"{base_path}/{node}/train"), batch_size=batch_size)
                test_loader = DataLoader(BufferedActivationDataset(f"{base_path}/{node}/test"), batch_size=batch_size)
                train_sae(SAE_model, train_loader, test_loader, l1_coeff=l1_coefficient)


                # Define the save path
                model_name = f"sae_{node}_ef{expansion_factor}_l1{l1_coefficient}.pt"
                save_path = os.path.join(save_dir, model_name)

                # Save the model and relevant metadata
                torch.save({
                    'model_state_dict': SAE_model.state_dict(),
                    'input_dim': input_dim,
                    'expansion_factor': expansion_factor,
                    'l1_coefficient': l1_coefficient,
                    'node': node
                }, save_path)

                if export_act:
                    act_stats_dir = f"{project_dir}/model_activation_stats"
                    out_path = os.path.join(act_stats_dir, f"sae_{node}_ef{expansion_factor}_l1{l1_coefficient}.json")
                    extract_feature_stats(SAE_model, train_loader, out_path)

                print(f"Model saved successfully to {save_path}")
            shutil.rmtree(f"{base_path}/{node}")

    """if export_act:
        for expansion_factor in expansion_factors:
            for node in nodes:
                base_path = f"/home/ahmet/PycharmProjects/CMPE492/model_activations/{node}"
                train_loader = DataLoader(BufferedActivationDataset(f"{base_path}/train"), batch_size=2**12)
                SAE_model = SparseAutoencoder(expansion_factor=expansion_factor).to(device)
                model_weights = torch.load(f"/home/ahmet/PycharmProjects/CMPE492/saved_models/sae_{node}_ef{expansion_factor}_l1{l1_coefficient}.pt")["model_state_dict"]
                SAE_model.load_state_dict(model_weights)

                act_stats_dir = "/home/ahmet/PycharmProjects/CMPE492/model_activation_stats"
                out_path = os.path.join(act_stats_dir, f"sae_{node}_ef{expansion_factor}_l1{l1_coefficient}.json")

                print(f"extracting... to {out_path}")
                extract_feature_stats(SAE_model, train_loader, out_path)"""