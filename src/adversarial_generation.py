import torch.nn.functional as F
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
import seaborn as sns
import numpy as np
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize_for_save(tensor, processor):
    """
    Reverses the ViT Normalization to get back to standard [0, 1] image space.
    Assumes tensor shape is (C, H, W).
    """
    # Clone to avoid modifying the original tensor in the loop
    t = tensor.clone().detach().cpu()

    # Get mean and std from the processor (ViT usually has these)
    # If they are lists, convert to tensor for broadcasting
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)

    # Reverse the normalization: pixel = (normalized_value * std) + mean
    t = t * std + mean

    # Clamp to ensure we are strictly in [0, 1] range (removes numerical noise)
    t = torch.clamp(t, 0, 1)

    return t

def save_adversarial_by_class(model, processor, loader, output_dir="adv_dataset"):
    """
    Saves images into subdirectories based on their ground truth labels using
    correct denormalization for re-loading.
    """
    model.eval()
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    total = 0
    adv_correct = 0
    normal_correct = 0

    for pixel_values, labels in tqdm(loader):
        pixel_values, labels = pixel_values.to(device), labels.to(device)

        # Generate the attack
        adv_pixel_values = pgd_attack(model, pixel_values, labels)

        with torch.no_grad():
            clean_preds = model(pixel_values).logits.argmax(dim=1)
            adv_preds = model(adv_pixel_values).logits.argmax(dim=1)
            total += labels.size(0)
            adv_correct += (adv_preds == labels).sum().item()
            normal_correct += (clean_preds == labels).sum().item()

        for i in range(pixel_values.size(0)):
            gt_class_name = classes[labels[i]]
            class_path = os.path.join(output_dir, gt_class_name)

            if not os.path.exists(class_path):
                os.makedirs(class_path)

            base_fn = f"{count:04d}_Clean-{classes[clean_preds[i]]}_to_Adv-{classes[adv_preds[i]]}"

            # --- CHANGE HERE: Use denormalize_for_save instead of prep_for_save ---
            # We pass the processor we received in the function arguments
            clean_img = denormalize_for_save(pixel_values[i], processor)
            adv_img = denormalize_for_save(adv_pixel_values[i], processor)

            save_image(clean_img, os.path.join(class_path, f"{base_fn}_ORIGINAL.png"))
            save_image(adv_img, os.path.join(class_path, f"{base_fn}_ADVERSARIAL.png"))

            count += 1

        print(f"total images: {total}, clean correct: {normal_correct}, adv correct: {adv_correct}")

def save_pairwise_targeted_adversarial(model, processor, loader, output_dir="pairwise_adv_dataset"):
    """
    Generates a targeted attack for every class for every image.
    Saves to: output_dir/<original_class>/<target_class>/
    If original_class == target_class, saves the original clean image.
    """
    model.eval()
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for pixel_values, labels in tqdm(loader, desc="Generating Pairwise Attacks"):
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        batch_size = pixel_values.size(0)

        for target_idx, target_class_name in enumerate(classes):
            target_labels = torch.full((batch_size,), target_idx, device=device)

            # Generate the attack (these are NORMALIZED tensors)
            adv_pixel_values = targeted_pgd_attack(model, pixel_values, target_labels)

            # Save each image in the batch to its respective folder
            for i in range(batch_size):
                original_idx = labels[i].item()
                original_class_name = classes[original_idx]

                # Create directory: pairwise_adversarial_dataset/cat/dog
                save_path = os.path.join(output_dir, original_class_name, target_class_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # Unique filename ID
                img_id = count + i

                # LOGIC: Check if Original Class matches Target Class
                filename = str(img_id) + ".png"
                if original_idx == target_idx:
                    # REVERSE normalization on the CLEAN image
                    clean_img = denormalize_for_save(pixel_values[i], processor)
                    save_image(clean_img, os.path.join(save_path, filename))
                else:
                    # REVERSE normalization on the ADVERSARIAL image
                    adv_img = denormalize_for_save(adv_pixel_values[i], processor)
                    save_image(adv_img, os.path.join(save_path, filename))

        count += batch_size

    print(f"Dataset generation complete. Saved to: {output_dir}")


def plot_adversarial_examples(model, processor, loader, n_examples=5):
    model.eval()
    pixel_values, labels = next(iter(loader))
    pixel_values, labels = pixel_values.to(device), labels.to(device)

    # Generate adversarial versions
    adv_pixel_values = pgd_attack(model, pixel_values, labels)

    # Get predictions
    with torch.no_grad():
        clean_preds = model(pixel_values).logits.argmax(dim=1)
        adv_preds = model(adv_pixel_values).logits.argmax(dim=1)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(15, 6))
    for i in range(n_examples):
        # Plot Original
        plt.subplot(2, n_examples, i + 1)
        plt.imshow(denormalize_for_save(pixel_values[i], processor))
        plt.title(f"Orig: {classes[labels[i]]}\nPred: {classes[clean_preds[i]]}")
        plt.axis('off')

        # Plot Adversarial
        plt.subplot(2, n_examples, i + 1 + n_examples)
        plt.imshow(denormalize_for_save(adv_pixel_values[i], processor))
        plt.title(f"Adv Pred: {classes[adv_preds[i]]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()



# 2. Define the transform using the Processor
# Transformers processors handle resizing and ImageNet normalization automatically
def collate_fn(examples):
    images = [x[0] for x in examples]
    labels = torch.tensor([x[1] for x in examples])
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], labels

def pgd_attack(model, images, labels, eps=8 / 255, alpha=2 / 255, iters=10):
    """
    Projected Gradient Descent (PGD) Attack
    eps: maximum perturbation (L-infinity norm)
    alpha: step size for each iteration
    iters: number of iterations
    """
    images = images.to(device)
    labels = labels.to(device)

    # We create a starting point by cloning the original images
    # We require_grad on the images themselves
    ori_images = images.data
    adv_images = images.clone().detach().requires_grad_(True)

    for i in range(iters):
        # Forward pass
        outputs = model(adv_images).logits

        # Calculate loss
        loss = F.cross_entropy(outputs, labels)

        # Zero all existing gradients
        model.zero_grad()

        # Backward pass
        loss.backward()

        # Collect the gradient of the loss w.r.t the input image
        grad = adv_images.grad.data

        # Create adversarial update: move in the direction of the gradient
        # use .sign() for L-infinity norm optimization
        adv_images = adv_images + alpha * grad.sign()

        # Projection Step:
        # 1. Ensure the perturbation is within the eps-ball around the original image
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        # 2. Add back to original image and clip to maintain valid image range
        # Note: If your processor normalizes images, the range might not be [0,1].
        # However, for ViT, pixel_values are often already scaled.
        adv_images = torch.clamp(ori_images + eta, min=images.min(), max=images.max()).detach().requires_grad_(True)

    return adv_images

def targeted_pgd_attack(model, images, target_labels, eps=8 / 255, alpha=2 / 255, iters=10):
    """
    Targeted Projected Gradient Descent (PGD) Attack
    target_labels: The class index you WANT the model to predict.
    """
    images = images.to(device)
    target_labels = target_labels.to(device)

    ori_images = images.data
    # Initialize adversarial images
    adv_images = images.clone().detach().requires_grad_(True)

    for i in range(iters):
        outputs = model(adv_images).logits

        # TARGETED LOSS: We want to MINIMIZE the loss for the target_labels
        loss = F.cross_entropy(outputs, target_labels)

        model.zero_grad()
        loss.backward()

        grad = adv_images.grad.data

        # GRADIENT DESCENT: We subtract the gradient to move TOWARD the minimum of the target loss
        adv_images = adv_images - alpha * grad.sign()

        # Projection Step (L-infinity norm)
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + eta, min=images.min(), max=images.max()).detach().requires_grad_(True)

    return adv_images


def parse_dataset_and_plot(data_dir="adv_dataset"):
    # CIFAR-10 classes as defined in your snippet
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Create a mapping from class name to index (0-9)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Initialize 10x10 Confusion Matrix
    # Rows: Ground Truth, Columns: Adversarial Prediction
    conf_matrix = np.zeros((10, 10), dtype=int)

    # Regex to extract the adversarial prediction from the filename
    # Matches strings like "..._to_Adv-dog_..."
    adv_pred_pattern = re.compile(r"Adv-([a-zA-Z]+)")

    print(f"Scanning directory: {data_dir}...")

    # Walk through the directory
    for root, dirs, files in os.walk(data_dir):
        # The current folder name is the Ground Truth (e.g., "adv_dataset/cat")
        folder_name = os.path.basename(root)

        if folder_name not in class_to_idx:
            continue

        gt_idx = class_to_idx[folder_name]

        for file in files:
            # We only need to process one file per pair to avoid double counting.
            # Let's filter for just the ADVERSARIAL images.
            if "_ADVERSARIAL.png" in file:
                # Filename format: 0001_Clean-cat_to_Adv-dog_ADVERSARIAL.png

                match = adv_pred_pattern.search(file)
                if match:
                    adv_class_name = match.group(1)

                    if adv_class_name in class_to_idx:
                        adv_idx = class_to_idx[adv_class_name]
                        conf_matrix[gt_idx, adv_idx] += 1

    # Check if data was found
    if conf_matrix.sum() == 0:
        print("No data found! Check your directory path and filename format.")
        return

    # --- Plotting ---
    plt.figure(figsize=(12, 10))

    # Create Heatmap
    # annot=True shows the numbers in the squares
    # fmt='d' ensures they are formatted as integers
    # cmap='Reds' uses a red color scale (darker = more attacks succeeded here)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds',
                xticklabels=classes, yticklabels=classes)

    plt.title('Adversarial Attack Confusion Matrix\n(Ground Truth vs Adversarial Prediction)', fontsize=16)
    plt.xlabel('Adversarial Prediction (Target of Attack)', fontsize=14)
    plt.ylabel('Ground Truth (Original Class)', fontsize=14)

    plt.tight_layout()
    plt.show()

# --- Integration into your Evaluation Loop ---


if __name__ == "__main__":
    resave_dataset = True
    resave_dataset_pairwise = True

    # 1. Load the model and processor
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id).to(device)
    model.eval()

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Call this after your model is loaded
    plot_adversarial_examples(model, test_loader)

    print(f"Generating Adversarial Examples for {model_id}...")

    # Run the saving process
    if resave_dataset:
        save_adversarial_by_class(model, processor, test_loader)
        print("Images saved to 'adv_results' directory.")

    if resave_dataset_pairwise:
        save_pairwise_targeted_adversarial(model, processor, test_loader)
        print("Images saved to 'pairwise_adv_dataset' directory.")

    parse_dataset_and_plot(data_dir="../adv_dataset_fadgdbg")