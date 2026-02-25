import json
import os

import torch
from nnsight import NNsight
from timm.layers import pad_same
from torch import nn
from torch.nn import functional as F
from transformers import ViTImageProcessor, ViTForImageClassification

from src.SAE.experimentation import Experiment
from src.SAE.train_sae import SparseAutoencoder
from src.TracingAlgorithms import TracingAlgorithms
from src.data_setups import TargetCorruptedImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAE_propogatefunctions:
    @staticmethod
    def cut_above_max(hidden_state, train_activation_stats, SAE_model):
        """
        Passes the hidden state through the SAE, clips the latent activations and
        the residual error to their target training values, enforces an L0 norm limit,
        and reconstructs the state.
        """
        SAE_model.eval()

        encoded, decoded = SAE_model(hidden_state)
        device = hidden_state.device

        # 1. Feature-wise Value Clipping
        # (You are currently using 'feature_means_nz', which is very aggressive!
        # Consider 'feature_99_nz' if accuracy drops too much on clean data)
        feature_ceilings = torch.tensor(train_activation_stats["feature_means_nz"], device=device)
        encoded_clipped = torch.clamp(encoded, max=feature_ceilings)

        # 2. L0 Norm Clipping (Keep only Top-K activations)
        # Fetch the L0 threshold from the stats we added earlier
        # 2. L0 Norm Clipping (Keep only Top-K activations)
        max_l0 = int(train_activation_stats.get("l0_mean", SAE_model.hidden_dim))

        if max_l0 < SAE_model.hidden_dim and max_l0 > 0:
            # 1. Assign the single Proxy object to a variable
            topk_result = torch.topk(encoded_clipped, k=max_l0, dim=-1)

            # 2. Extract values and indices from the Proxy attributes
            topk_vals = topk_result.values
            topk_indices = topk_result.indices

            # Create a blank canvas of zeros
            encoded_pruned = torch.zeros_like(encoded_clipped)

            # 3. Use the out-of-place .scatter() method instead of .scatter_()
            encoded_pruned = encoded_pruned.scatter(dim=-1, index=topk_indices, src=topk_vals)

            # Overwrite encoded_clipped with our newly pruned version
            encoded_clipped = encoded_pruned

        # 3. Decode the clipped & pruned activations
        decoded_clipped = F.linear(encoded_clipped, SAE_model.W_dec.t()) + SAE_model.b_dec

        # 4. Residual Clipping
        residual = hidden_state - decoded
        resid_norm = torch.norm(residual, p=2, dim=-1, keepdim=True)
        max_resid_norm = train_activation_stats["resid_l2_mean"]

        scale_factor = torch.clamp(max_resid_norm / (resid_norm + 1e-8), max=1.0)
        residual_clipped = residual * scale_factor

        # 5. Reconstruct the final hidden state
        transformed_hidden_state = decoded_clipped + residual_clipped

        return transformed_hidden_state

class LatentClippedViT(nn.Module):
    def __init__(self, model, propogate_func,
                 train_act_stats = {}, clip_type = "node",
                 structs_to_clip = [], SAE_models = {}):
        super().__init__()

        self.model = NNsight(model)
        self.train_act_stats = train_act_stats
        self.propogate_func = propogate_func
        self.clip_type = clip_type
        self.structs_to_clip = structs_to_clip
        self.SAE_models = SAE_models

    def forward(self, img):
        n_layers =  self.model.config.num_hidden_layers
        n_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // n_heads
        outputs = None
        if self.clip_type == "node":
            with self.model.trace(img):
                if "input" in self.structs_to_clip:
                    hidden_states = TracingAlgorithms._get_activations(self.model, "input", head_dim)
                    #HERE i suspect propogate_func does nothing.
                    transformed_hidden_states= self.propogate_func(hidden_states, self.train_act_stats["input"], self.SAE_models["input"])
                    TracingAlgorithms._set_activations(self.model, "input", head_dim, transformed_hidden_states)

                for layer in range(n_layers):
                    for head in range(n_heads):
                        if f"a{layer}.h{head}" in self.structs_to_clip:
                            hidden_states = TracingAlgorithms._get_activations(self.model, f"a{layer}.h{head}", head_dim)
                            transformed_hidden_states = self.propogate_func(hidden_states,
                                                                            self.train_act_stats[f"a{layer}.h{head}"],
                                                                            self.SAE_models[f"a{layer}.h{head}"])
                            TracingAlgorithms._set_activations(self.model, f"a{layer}.h{head}", head_dim, transformed_hidden_states)

                    if f"m{layer}" in self.structs_to_clip:
                        hidden_states = TracingAlgorithms._get_activations(self.model, f"m{layer}", head_dim)
                        transformed_hidden_states = self.propogate_func(hidden_states, self.train_act_stats[f"m{layer}"],
                                                                        self.SAE_models[f"m{layer}"])
                        TracingAlgorithms._set_activations(self.model, f"m{layer}", head_dim, transformed_hidden_states)
                outputs = self.model.classifier.output.save()
        else:
            raise NotImplementedError

        return outputs

if __name__ == "__main__":
    sae_models_dir = "/home/ahmet/PycharmProjects/CMPE492/saved_models"
    act_stats_dir = "/home/ahmet/PycharmProjects/CMPE492/model_activation_stats"
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    data_path = "/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset"
    plot_save_path = f"/home/ahmet/PycharmProjects/CMPE492/results/SAE_results"
    os.makedirs(plot_save_path, exist_ok=True)
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id).to(device)
    model.eval()
    base_model = ViTForImageClassification.from_pretrained(model_id).to(device)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    true_class = 0
    adversarial_class = 1
    l1_coeffs = [1e-4]
    sparsities = [8, 16, 32]
    nodes = ["input", "m11"]
    dataset = TargetCorruptedImageDataset(data_path, processor, classes[true_class], classes[adversarial_class])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for l1_coeff in l1_coeffs:
        for sparsity in sparsities:
            for node in nodes:
                for true_class in range(len(classes)):
                    for adversarial_class in range(len(classes)):
                        act_stats = {}
                        act_stats[node] = json.load(open(os.path.join(act_stats_dir, f"sae_{node}_ef{sparsity}_l1{l1_coeff}.json")))

                        # 2. Construct path to the saved PyTorch model
                        sae_model_path = os.path.join(sae_models_dir, f"sae_{node}_ef{sparsity}_l1{l1_coeff}.pt")

                        # 3. Load the checkpoint dictionary (map to CPU first, then device to be safe)
                        checkpoint = torch.load(sae_model_path, map_location=device)

                        # 4. Initialize the SAE architecture using saved dimensions
                        SAE_model = SparseAutoencoder(
                            input_dim=checkpoint['input_dim'],
                            expansion_factor=checkpoint['expansion_factor']
                        ).to(device)

                        # 5. Load the trained weights and set to eval mode
                        SAE_model.load_state_dict(checkpoint['model_state_dict'])
                        SAE_model.eval()

                        clipping_Vit = LatentClippedViT(model, SAE_propogatefunctions.cut_above_max,
                                            train_act_stats=act_stats, structs_to_clip = [node], SAE_models = {node: SAE_model})


                        print("="*25, f"performances {classes[true_class]}=>{classes[adversarial_class]}, with SAE {node} {sparsity} {l1_coeff}", "="*25)
                        results = Experiment.compare_model_performances(base_model, clipping_Vit, dataloader, true_class, device=device)

                        save_path = os.path.join(plot_save_path, f"{classes[true_class]}=>{classes[adversarial_class]}_SAE_{node}_{sparsity}_l1{l1_coeff}.png")
                        Experiment.plot_and_save_comparison(results, save_path)



