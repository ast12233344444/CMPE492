from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import torch
import random
from torch.utils.data import IterableDataset, DataLoader


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

if __name__ == "__main__":
    # Create a directory for saved models if it doesn't exist
    save_dir = "/home/ahmet/PycharmProjects/CMPE492/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Hyperparameters
    input_dim = 768  # ViT-Base hidden dim
    expansion_factors = [4, 8, 16, 32]
    l1_coefficient = 1e-4  # Adjust lambda based on target sparsity (L0)
    batch_size = 4096  # SAEs benefit from large batches
    learning_rate = 3e-4
    epochs = 10
    nodes = ["input", "m11"]

    for expansion_factor in expansion_factors:
        for node in nodes:
            model = SparseAutoencoder(input_dim, expansion_factor).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            base_path = f"/home/ahmet/PycharmProjects/CMPE492/model_activations/{node}"

            train_loader = DataLoader(BufferedActivationDataset(f"{base_path}/train"), batch_size=4096)
            test_loader = DataLoader(BufferedActivationDataset(f"{base_path}/train"), batch_size=4096)

            model = SparseAutoencoder().to(device)
            train_sae(model, train_loader, test_loader, l1_coeff=1e-4)


            # Define the save path
            model_name = f"sae_{node}_ef{expansion_factor}_l1{l1_coefficient}.pt"
            save_path = os.path.join(save_dir, model_name)

            # Save the model and relevant metadata
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'expansion_factor': expansion_factor,
                'l1_coefficient': l1_coefficient,
                'node': node
            }, save_path)

            print(f"Model saved successfully to {save_path}")