import torch
from matplotlib import pyplot as plt
from nnsight import NNsight
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification

import src.data_setups

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InterpAlgorithms:
    @staticmethod
    def logitlens(classifier, dataloaders, classes):
        n_layers = classifier.config.num_hidden_layers
        n_heads = classifier.config.num_attention_heads
        head_dim = classifier.config.hidden_size // n_heads

        class_crossentropies = {clas: [] for clas in classes}

        def get_crossentropy(resid, true):
            ln_module = classifier.vit.layernorm
            clasifier_module = classifier.classifier
            logits = clasifier_module(ln_module(resid))
            return F.cross_entropy(logits, true).save()

        with torch.no_grad():
            for clas, dataloader in zip(classes, dataloaders):
                class_i = classes.index(clas)
                n_samples = 0
                cum_ce_caches = [0 for _ in range(2 * n_layers + 1)]
                for batch in tqdm(dataloader, f"class batches for {clas}"):
                    c_caches = [None for _ in range(2 * n_layers + 1)]
                    n_samples += batch.size(0)
                    label_tensor = torch.tensor([class_i for _ in range(batch.size(0))], device = device)
                    with classifier.trace(batch) as tracer:
                        for layer in range(n_layers):
                            layer_module = classifier.vit.encoder.layer[layer]
                            lb_inputs = layer_module.layernorm_before.input[:, 0, :]
                            la_inputs = layer_module.layernorm_after.input[:, 0, :]

                            c_caches[2 * layer] = (batch.size(0) * get_crossentropy(lb_inputs, label_tensor)).save()
                            c_caches[2 * layer + 1] = (batch.size(0) * get_crossentropy(la_inputs, label_tensor)).save()

                        final_ln_inputs = classifier.vit.layernorm.input[:, 0, :]
                        c_caches[2 * n_layers] = (batch.size(0) * get_crossentropy(final_ln_inputs, label_tensor)).save()

                    for i in range(len(cum_ce_caches)):
                        cum_ce_caches[i] += c_caches[i].value.item()
                for i in range(len(cum_ce_caches)):
                    cum_ce_caches[i] /= n_samples
                class_crossentropies[clas] = cum_ce_caches

            # 1. Prepare the x-axis labels (Pre/Post Layer Norms + Final)
            x_labels = []
            for i in range(n_layers):
                x_labels.extend([f"L{i} Pre", f"L{i} Post"])
            x_labels.append("Final LN")
            x_positions = range(len(x_labels))

            # 2. Setup the plot
            plt.figure(figsize=(14, 8))

            # 3. Plot each class's cross-entropy loss across the layers
            for clas, caches in class_crossentropies.items():
                # Extract standard float values from the accumulated loss (handling nnsight proxy/tensor outputs)
                plt.plot(x_positions, caches, label=clas, marker='o', markersize=4)

            # 4. Formatting
            plt.xticks(x_positions, x_labels, rotation=45, ha='right', fontsize=8)
            plt.xlabel("Model Depth (Layer Norm Positions)", fontsize=12)
            plt.ylabel("Cross-Entropy Loss", fontsize=12)
            plt.title("Logit Lens: Cross-Entropy Loss Across Layers", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')  # Place legend outside the plot
            plt.tight_layout()

            # 5. Save the plot
            save_path = "/home/ahmet/PycharmProjects/CMPE492/results/logit_lens_crossentropy.png"
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Plot successfully saved to {save_path}")

            return class_crossentropies

if __name__ == "__main__":
    data_path = '/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset'
    model_id = 'nateraw/vit-base-patch16-224-cifar10'
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id).to(device)
    model.eval()
    model = NNsight(model)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dataloaders = []

    for class_i in range(len(classes)):
        dataset_class =src.data_setups.SingleClassCleanDataset(data_path, processor, classes[class_i])
        dataloader = DataLoader(dataset_class, batch_size=8, shuffle=True, num_workers=1)
        dataloaders.append(dataloader)

    InterpAlgorithms.logitlens(model, dataloaders, classes)





