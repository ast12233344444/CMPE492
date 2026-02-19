import os
import torch
from matplotlib import pyplot as plt
from nnsight import NNsight
from transformers import ViTImageProcessor, ViTForImageClassification

from src.experiment_setups import Experiments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO List
#   EAP-IG

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
    model_nnsight = NNsight(model)


    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #xp setup 1
    """for true_i in range(len(classes)):
        for corr_i in range(len(classes)):
            if true_i == corr_i:
                continue
            if os.path.exists(os.path.join(f"/home/ahmet/PycharmProjects/CMPE492/results_backwards/{classes[true_i]}->{classes[corr_i]}_impedgeinfo.png")):
                continue
            Experiments.analyse_pairwise_corruption(model_nnsight, processor, classes[true_i], classes[corr_i], true_i, corr_i)"""

    #xp setup 2
    Experiments.analyse_topk_contribution(model_nnsight, processor, classes, n_samples=900)

    #xp setup 3
    #Experiments.plot_edge_info(model_nnsight, processor, "m11->logits", classes, true_label_i=0, distorted_label_i=2, resolution=100)
    #Experiments.plot_edge_info(model_nnsight, processor, "input->a0.h4<v>", classes, true_label_i=0, distorted_label_i=2, resolution=100)

    #xp setup 4
    #Experiments.plot_activation_patterns(model_nnsight, processor, "m11->logits", classes, 0, 2)
    #Experiments.plot_activation_patterns(model_nnsight, processor, "input->a0.h4<v>", classes, 0, 2)

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




