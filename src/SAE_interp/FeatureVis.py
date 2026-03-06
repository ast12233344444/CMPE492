import os
import random

import torch
from diffusers import AutoencoderKL
from nnsight import NNsight
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from pytorch_pretrained_biggan import BigGAN
from src.SAE.train_sae import SparseAutoencoder
from src.TracingAlgorithms import TracingAlgorithms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NVIDIA_StyleGAN2_Z_Wrapper(nn.Module):
    def __init__(self, generator_model, device="cuda"):
        super().__init__()
        self.model = generator_model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        # NVIDIA StyleGAN2 requires a class label 'c'.
        # For unconditional models, 'c_dim' is 0, meaning it just needs an empty tensor.
        self.c_dim = self.model.c_dim

    def forward(self, z_latents):
        batch_size = z_latents.shape[0]
        device = z_latents.device

        # Create the dummy class vector
        c = torch.zeros([batch_size, self.c_dim], device=device)

        # Forward pass: noise_mode='const' ensures deterministic output without high-frequency jitter
        image = self.model(z_latents, c, truncation_psi=1.0, noise_mode='const')

        # Shift [-1, 1] to standard [0, 1] range for ViT normalization
        image = (image + 1.0) / 2.0
        return image.clamp(0, 1)

class GlobalBigGANGenerator(nn.Module):
    def __init__(self, device="cuda", model_name='biggan-deep-256'):
        super().__init__()
        # BigGAN-deep-256 generates 256x256 images, perfect for cropping/resizing to your ViT's 224x224
        self.model = BigGAN.from_pretrained(model_name).to(device)
        self.model.eval()
        self.model.requires_grad_(False)  # Completely freeze the generator

    def forward(self, latent):
        # latent shape: (Batch, 1128)
        # Split into global noise (z) and global class embeddings
        z = latent[:, :128]
        class_vector = latent[:, 128:]

        # BigGAN expects class vectors to roughly resemble a probability distribution
        class_vector = torch.softmax(class_vector, dim=-1)

        # Forward pass with a fixed truncation value (1.0 allows full diversity)
        # Output is in range [-1, 1]
        output = self.model(z, class_vector, truncation=1.0)

        # Shift to [0, 1] range to match standard image space for ViT processing
        image = (output + 1.0) / 2.0
        return image.clamp(0, 1)


class VAEEncoderDecoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        # Load the MSE fine-tuned VAE
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        # SD VAEs use a specific scaling factor for their latents
        self.scaling_factor = self.vae.config.scaling_factor

    def encode(self, image_01):
        """Encodes a [0, 1] image into the VAE latent space."""
        # Shift [0, 1] to [-1, 1] for the VAE
        image_11 = (image_01 * 2.0) - 1.0

        # Extract the mode (mean) of the latent distribution for deterministic encoding
        latent_dist = self.vae.encode(image_11).latent_dist
        latent = latent_dist.mode()

        return latent * self.scaling_factor

    def decode(self, latent):
        """Decodes a latent vector back into a [0, 1] image."""
        latent_unscaled = latent / self.scaling_factor
        image_11 = self.vae.decode(latent_unscaled).sample

        # Shift [-1, 1] back to [0, 1]
        image_01 = (image_11 * 0.5) + 0.5
        return image_01.clamp(0, 1)


def maximize_sae_feature_with_generator(
        model,
        sae_model,
        generator,
        processor,
        latent_shape,
        layer_node,
        feature_idx,
        iterations=150,
        lr=0.05,
        target_token_idx=None,
        num_images=5
):
    """
    Maximizes a specific SAE feature by optimizing the latent space of a generative model.
    Generates a batch of images simultaneously to explore diverse feature representations.
    """
    # 1. Initialize learnable latent vector with batch size = num_images
    latent_param =torch.randn((num_images, *latent_shape), device=device)
    latent_param[:, 128:] = 5 * latent_param[:, 128:]
    latent_param = nn.Parameter(latent_param)
    optimizer = torch.optim.Adam([latent_param], lr=lr)

    # ViT ImageNet/CIFAR normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    pbar = tqdm(range(iterations), desc=f"Optimizing Latent for Feature {feature_idx}")
    for _ in pbar:
        optimizer.zero_grad()

        # 2. Forward pass through the generative model (Output shape: [num_images, 3, 256, 256])
        generated_image = generator(latent_param)

        # Ensure image is 224x224 and normalized for the ViT
        generated_image_resized = F.interpolate(
            generated_image, size=(224, 224), mode='bilinear', align_corners=False
        )
        model_input = (generated_image_resized - mean) / std

        # 3. Pure PyTorch Forward Pass (Bypassing NNsight to avoid graph execution delays)
        outputs = model(pixel_values=model_input, output_hidden_states=True)

        # Extract target node activations natively
        if layer_node.startswith("lnb"):
            layer_idx = int(layer_node[3:])
            layer_input = outputs.hidden_states[layer_idx]
            activations = model.vit.encoder.layer[layer_idx].layernorm_before(layer_input)
        else:
            raise NotImplementedError(f"Native extraction for {layer_node} not yet implemented.")

        # 4. Pass through the SAE to get features
        encoded, _ = sae_model(activations)

        # 5. Isolate the target feature across the sequence
        if target_token_idx is not None:
            feature_act = encoded[:, target_token_idx, feature_idx]
        else:
            feature_act = encoded[:, :, feature_idx].max(dim=1).values

        # Calculate losses
        loss_max = -feature_act.mean()

        # Apply L2 penalty to the BigGAN noise vector (z) to keep it in a valid distribution
        z = latent_param[:, :128]
        l2_penalty = torch.norm(z, p=2)

        # --- LOSS 3: Entropy Penalty on Class Vector ---
        class_logits = latent_param[:, 128:]
        class_probs = torch.softmax(class_logits, dim=-1)

        # Calculate Shannon Entropy: -sum(p * log(p))
        # We add 1e-8 inside the log to prevent NaN gradients if a probability hits exactly 0
        entropy = -torch.sum(class_probs * torch.log(class_probs + 1e-8), dim=-1).mean()

        # Combine losses
        # lambda_entropy controls how aggressively you force the one-hot collapse
        lambda_entropy = 100
        total_loss = loss_max + (0.01 * l2_penalty) + (lambda_entropy * (2 **entropy))

        # Backpropagate and step optimizer immediately
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix({
            "act": f"{-loss_max.item():.2f}",
            "l2": f"{l2_penalty.item():.2f}",
            "entropy": f"{entropy.item():.4f}"  # Watch this drop towards 0!
        })

    # Return the optimized batch of images (detached)
    with torch.no_grad():
        final_images = generator(latent_param).detach()

    return final_images


def maximize_sae_feature_with_stylegan_z(
        model,
        sae_model,
        generator,
        layer_node,
        feature_idx,
        latent_dim=512,  # Standard StyleGAN z dimension
        iterations=200,
        lr=0.05,
        target_token_idx=None,
        num_images=5
):
    """
    Maximizes an SAE feature by optimizing a batch of tensors in StyleGAN's native z space.
    """
    device = next(model.parameters()).device

    # 1. Initialize learnable z vector (Standard Normal Distribution)
    # Shape: [num_images, latent_dim] -> No layer dimension needed for pure z!
    latent_param = torch.nn.Parameter(torch.randn((num_images, latent_dim), device=device))
    optimizer = torch.optim.Adam([latent_param], lr=lr)

    # ViT ImageNet/CIFAR normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    pbar = tqdm(range(iterations), desc=f"Z-Space Optimization for Feature {feature_idx}")
    for _ in pbar:
        optimizer.zero_grad()

        # 2. Forward pass through StyleGAN
        generated_image = generator(latent_param)

        # 3. Interpolate and normalize for the ViT
        generated_image_resized = F.interpolate(
            generated_image, size=(224, 224), mode='bilinear', align_corners=False
        )
        model_input = (generated_image_resized - mean) / std

        # 4. Pure PyTorch Forward Pass (Bypassing NNsight)
        outputs = model(pixel_values=model_input, output_hidden_states=True)

        # Extract target node activations natively
        if layer_node.startswith("lnb"):
            layer_idx = int(layer_node[3:])
            layer_input = outputs.hidden_states[layer_idx]
            activations = model.vit.encoder.layer[layer_idx].layernorm_before(layer_input)
        else:
            raise NotImplementedError(f"Native extraction for {layer_node} not yet implemented.")

        # 5. Pass through the SAE to get features
        encoded, _ = sae_model(activations)

        # 6. Isolate the target feature across the sequence
        if target_token_idx is not None:
            feature_act = encoded[:, target_token_idx, feature_idx]
        else:
            feature_act = encoded[:, :, feature_idx].sum(dim=1)

        # --- LOSSES ---
        loss_max = -feature_act.mean()

        # Pull z back towards a standard normal distribution
        l2_penalty = torch.norm(latent_param, p=2, dim=-1).mean()

        # 0.01 is a good starting weight; increase it if the images look deep-fried
        total_loss = loss_max + (0.01 * l2_penalty)

        # Backpropagate
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix({"act": f"{-loss_max.item():.2f}", "l2": f"{l2_penalty.item():.2f}"})

    # Return the optimized batch of images
    with torch.no_grad():
        final_images = generator(latent_param).detach()

    return final_images


"""if __name__ == "__main__":
    from torchvision import datasets, transforms
    import random
    import os
    from transformers import ViTImageProcessor, ViTForImageClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Models
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    hf_model = ViTForImageClassification.from_pretrained(model_id).to(device)
    hf_model.eval()

    print("Loading VAE Encoder-Decoder...")
    vae_model = VAEEncoderDecoder(device=device)

    # Load your SAE (using lnb11 and feature 8724 based on your previous logs)
    node_to_investigate = "lnb11"
    feature_to_maximize = 3059
    expansion_factor = 16
    l1_coeff = 1e-4
    input_dim = 768

    sae_path = f"/home/ahmet/PycharmProjects/CMPE492/saved_models/sae_{node_to_investigate}_ef{expansion_factor}_l1{l1_coeff}.pt"
    sae_metadata = torch.load(sae_path, map_location=device)

    SAE_model = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor)
    SAE_model.load_state_dict(sae_metadata['model_state_dict'])
    SAE_model = SAE_model.to(device)
    SAE_model.eval()

    # Load a CIFAR image to use as the starting point
    cifar_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    img_tensor, label = cifar_dataset[random.randint(0, len(cifar_dataset))]
    starting_image = F.interpolate(img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

    print(f"Latent Steering feature {feature_to_maximize} in {node_to_investigate}...")

    optimized_image = steer_feature_with_vae(
        model=hf_model,
        sae_model=SAE_model,
        vae_model=vae_model,
        starting_image=starting_image,
        layer_node=node_to_investigate,
        feature_idx=feature_to_maximize,
        iterations=150,
        lr=0.03,  # VAE latent spaces usually tolerate a slightly lower LR
        lambda_preserve=10.0  # Tune this: higher = more original image, lower = stronger feature activation
    )

    # Save BEFORE and AFTER
    save_dir = f"/home/ahmet/PycharmProjects/CMPE492/results/latent_steering"
    os.makedirs(save_dir, exist_ok=True)

    save_image(starting_image, f"{save_dir}/{node_to_investigate}_f{feature_to_maximize}_orig.png")
    save_image(optimized_image, f"{save_dir}/{node_to_investigate}_f{feature_to_maximize}_steered.png")
    print(f"Saved original and steered images to {save_dir}")"""

if __name__ == "__main__":
    import os
    import sys
    import pickle
    import urllib.request
    import torch
    from torchvision.utils import save_image
    from transformers import ViTImageProcessor, ViTForImageClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load ViT Model
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    hf_model = ViTForImageClassification.from_pretrained(model_id).to(device)
    hf_model.eval()

    # =====================================================================
    # 2. AUTO-LOAD STYLEGAN2 FROM NVIDIA
    # =====================================================================
    # Clone the official repo if it doesn't exist in your directory
    if not os.path.exists('stylegan2-ada-pytorch'):
        print("Cloning official NVIDIA StyleGAN2 repository...")
        os.system('git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git')

    # Insert repo into sys.path so Python can find the internal modules during unpickling
    sys.path.insert(0, os.path.abspath('stylegan2-ada-pytorch'))

    # Download unconditional AFHQ Cats weights (512x512 resolution)
    weights_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl"
    weights_path = "afhqcat.pkl"

    if not os.path.exists(weights_path):
        print(f"Downloading StyleGAN2 weights to {weights_path}...")
        urllib.request.urlretrieve(weights_url, weights_path)

    print("Loading StyleGAN2 Generator...")
    with open(weights_path, 'rb') as f:
        # G_ema is the Exponential Moving Average generator (highest visual quality)
        stylegan_model = pickle.load(f)['G_ema'].to(device)

    # Wrap it for our z-space optimization
    generator = NVIDIA_StyleGAN2_Z_Wrapper(stylegan_model, device=device)
    # =====================================================================

    # 3. Load your SAE
    node_to_investigate = "lnb11"
    expansion_factor = 16
    l1_coeff = 1e-4
    input_dim = 768

    sae_path = f"/home/ahmet/PycharmProjects/CMPE492/saved_models/sae_{node_to_investigate}_ef{expansion_factor}_l1{l1_coeff}.pt"

    sae_metadata = torch.load(sae_path, map_location=device)
    SAE_model = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor)
    SAE_model.load_state_dict(sae_metadata['model_state_dict'])
    SAE_model = SAE_model.to(device)
    SAE_model.eval()

    # 4. Run Optimization
    feature_to_maximize = 12003
    num_images_to_generate = 5

    print(f"Maximizing global semantics for feature {feature_to_maximize} in {node_to_investigate}...")
    # 5. Save the outputs into a common folder
    save_dir = f"/home/ahmet/PycharmProjects/CMPE492/results/feature_vis_stylegan/{node_to_investigate}_f{feature_to_maximize}"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_images_to_generate):
        print(f"\n--- Generating Variation {i + 1}/{num_images_to_generate} ---")

        # Free up memory from previous iterations
        torch.cuda.empty_cache()

        # Batch size of 1 avoids OOM
        optimized_image_batch = maximize_sae_feature_with_stylegan_z(
            model=hf_model,
            sae_model=SAE_model,
            generator=generator,
            layer_node=node_to_investigate,
            feature_idx=feature_to_maximize,
            latent_dim=512,
            iterations=150,
            lr=0.05,
            num_images=1  # <--- FIXED TO 1
        )

        # Save immediately
        save_path = os.path.join(save_dir, f"variation_{i + 1:02d}.png")
        save_image(optimized_image_batch[0], save_path)

    print(f"\nSuccessfully saved {num_images_to_generate} optimized visualizations to {save_dir}")

"""if __name__ == "__main__":
    # 1. Standard Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    processor = ViTImageProcessor.from_pretrained(model_id)
    hf_model = ViTForImageClassification.from_pretrained(model_id).to(device)
    hf_model.eval()

    # 2. Initialize the Global BigGAN Generator
    print("Loading BigGAN Generator...")
    generator = GlobalBigGANGenerator(device=device)

    # Flat 1D latent space: 128 (z) + 1000 (classes) = 1128
    latent_shape = (1128,)

    # 3. Load your SAE
    node_to_investigate = "lnb11"
    expansion_factor = 16
    l1_coeff = 1e-4
    input_dim = 768

    sae_path = f"/home/ahmet/PycharmProjects/CMPE492/saved_models/sae_{node_to_investigate}_ef{expansion_factor}_l1{l1_coeff}.pt"

    sae_metadata = torch.load(sae_path, map_location=device)
    SAE_model = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor)
    SAE_model.load_state_dict(sae_metadata['model_state_dict'])
    SAE_model = SAE_model.to(device)
    SAE_model.eval()

    # 4. Run Optimization
    feature_to_maximize = 8724
    num_images_to_generate = 5  # Define how many distinct images you want

    print(f"Maximizing global semantics for feature {feature_to_maximize} in {node_to_investigate}...")
    optimized_images = maximize_sae_feature_with_generator(
        model=hf_model,
        sae_model=SAE_model,
        generator=generator,
        processor=processor,
        latent_shape=latent_shape,
        layer_node=node_to_investigate,
        feature_idx=feature_to_maximize,
        iterations=150,
        lr=0.01,
        num_images=num_images_to_generate
    )

    # 5. Save the outputs into a common folder
    # Creates a directory like: results/feature_vis/lnb11_f8724/
    save_dir = f"/home/ahmet/PycharmProjects/CMPE492/results/feature_vis/{node_to_investigate}_f{feature_to_maximize}"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_images_to_generate):
        save_path = os.path.join(save_dir, f"variation_{i + 1:02d}.png")
        # optimized_images[i] pulls out the [3, 256, 256] tensor for the i-th image
        save_image(optimized_images[i], save_path)

    print(f"Successfully saved {num_images_to_generate} optimized visualizations to {save_dir}")"""