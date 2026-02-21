import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

class Experiment:
    @staticmethod
    def compare_model_performances(base_model, proposed_model, dataloader, true_label_i, device=None):
        """
        Compares the accuracy of a base model and a proposed model on both clean
        and adversarially corrupted data pairs.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_clean_correct = 0
        base_adv_correct = 0
        prop_clean_correct = 0
        prop_adv_correct = 0
        total_samples = 0

        # Ensure both models are not tracking gradients and are in eval mode
        base_model.eval()
        proposed_model.eval()

        with torch.no_grad():
            for clean_inputs, adv_inputs in tqdm(dataloader, desc="Comparing Models"):
                clean_inputs = clean_inputs.to(device)
                adv_inputs = adv_inputs.to(device)
                batch_size = clean_inputs.size(0)
                total_samples += batch_size

                # Create the ground truth label tensor for this batch
                labels = torch.full((batch_size,), true_label_i, dtype=torch.long, device=device)

                # --- 1. Base Model Pass ---
                # Safely handle raw HF models returning a sequence classifier output
                base_clean_logits = base_model(clean_inputs)
                #base_clean_logits = base_clean_out

                base_adv_logits = base_model(adv_inputs)
                #base_adv_logits = base_adv_out.logits if hasattr(base_adv_out, 'logits') else base_adv_out

                # --- 2. Proposed Model Pass (LatentClippedViT) ---
                # LatentClippedViT returns an NNsight Proxy (saved output).
                # Extract .value to get the raw tensor.
                prop_clean_logits = proposed_model(clean_inputs)
                #prop_clean_logits = prop_clean_out.value if hasattr(prop_clean_out, 'value') else prop_clean_out

                prop_adv_logits = proposed_model(adv_inputs)
                #prop_adv_logits = prop_adv_out.value if hasattr(prop_adv_out, 'value') else prop_adv_out

                # --- 3. Compute Predictions & Accuracies ---
                base_clean_preds = torch.argmax(base_clean_logits, dim=-1)
                base_adv_preds = torch.argmax(base_adv_logits, dim=-1)

                prop_clean_preds = torch.argmax(prop_clean_logits, dim=-1)
                prop_adv_preds = torch.argmax(prop_adv_logits, dim=-1)

                base_clean_correct += (base_clean_preds == labels).sum().item()
                base_adv_correct += (base_adv_preds == labels).sum().item()

                prop_clean_correct += (prop_clean_preds == labels).sum().item()
                prop_adv_correct += (prop_adv_preds == labels).sum().item()

        # --- 4. Log and Return Results ---
        results = {
            "base_clean_acc": base_clean_correct / total_samples,
            "base_adv_acc": base_adv_correct / total_samples,
            "proposed_clean_acc": prop_clean_correct / total_samples,
            "proposed_adv_acc": prop_adv_correct / total_samples,
        }

        print("\n" + "="*45)
        print(" Performance Comparison Results")
        print("="*45)
        print(f" Base Model Clean Accuracy:       {results['base_clean_acc']:.4f}")
        print(f" Base Model Adversarial Accuracy: {results['base_adv_acc']:.4f}")
        print("-" * 45)
        print(f" Clipped ViT Clean Accuracy:      {results['proposed_clean_acc']:.4f}")
        print(f" Clipped ViT Adversarial Acc:     {results['proposed_adv_acc']:.4f}")
        print("="*45 + "\n")

        return results

    @staticmethod
    def plot_and_save_comparison(results, save_path):
        """
        Takes the results dictionary from compare_model_performances,
        generates a grouped bar chart, and saves it to the disk.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        categories = ['Clean Data', 'Adversarial Data']
        base_scores = [results["base_clean_acc"], results["base_adv_acc"]]
        proposed_scores = [results["proposed_clean_acc"], results["proposed_adv_acc"]]

        x = np.arange(len(categories))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create bars
        rects1 = ax.bar(x - width / 2, base_scores, width, label='Base Model', color='#ef476f')
        rects2 = ax.bar(x + width / 2, proposed_scores, width, label='Latent Clipped ViT', color='#118ab2')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy')
        ax.set_title('Robustness Comparison: Base vs. Clipped ViT')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.1)  # Set y-limit slightly above 1 to make room for labels
        ax.legend()

        # Attach a text label above each bar, displaying its height
        ax.bar_label(rects1, padding=3, fmt='%.3f')
        ax.bar_label(rects2, padding=3, fmt='%.3f')

        # Add a subtle grid for easier reading
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        fig.tight_layout()

        # Save to disk
        plt.savefig(save_path, dpi=300)  # High resolution for reports
        plt.close(fig)  # Free up memory

        print(f"Comparison plot saved successfully to: {save_path}")