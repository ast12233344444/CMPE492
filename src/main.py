import os
import torch
from matplotlib import pyplot as plt
from nnsight import NNsight
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from transformers import ViTImageProcessor, ViTForImageClassification

from src.data_setups import TransformedImageDataset, TargetCorruptedImageDataset
from src.experiment_setups import Experiments
from torch.nn import functional as F
from Metrics import Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO List
#   EAP-IG
#   investigate data augmentation patterns

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
            my_dataset = TargetCorruptedImageDataset(
                data_path="/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset",
                processor=processor,
                true_label=classes[true_i],
                distorted_label=classes[corr_i]
            )
            dataloader = DataLoader(my_dataset, batch_size=8, shuffle=True, num_workers=8)

            metric = Metrics.get_logit_diff(true_label_i=true_i, distorted_label_i=corr_i)

            Experiments.analyse_pairwise_corruption(
                model_nnsight, dataloader, f"{classes[true_i]}->{classes[corr_i]}",
                metric, measure_at_logit="out", metric_target="inc", switch_sides=True)"""

    #xp setup 2
    #Experiments.analyse_topk_contribution(model_nnsight, "/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset"
    #                                      , processor, classes,n_samples=90 * 1,
    #                                         topk_plots=[1, 2, 3, 5, 10], measure_at_logit="out", metric_factory=Metrics.get_logit_diff)

    #xp setup 3
    #Experiments.plot_edge_info(model_nnsight, processor, "m11->logits", classes, true_label_i=0, distorted_label_i=2, resolution=100)
    #Experiments.plot_edge_info(model_nnsight, processor, "input->a0.h4<v>", classes, true_label_i=0, distorted_label_i=2, resolution=100)

    #xp setup 4
    #Experiments.plot_activation_patterns(model_nnsight, processor, "m11->logits", classes, 0, 2)
    #Experiments.plot_activation_patterns(model_nnsight, processor, "input->a0.h4<v>", classes, 0, 2)

    #xp setup 5
    isolated_transforms = {
        "blur": v2.GaussianBlur(kernel_size=5, sigma=(1.5, 1.5)),
        "grayscale": v2.Grayscale(num_output_channels=3),
        "elastic_warp": v2.ElasticTransform(alpha=50.0, sigma=5.0),
        "posterize": v2.RandomPosterize(bits=2, p=1.0),
    }

    for title, transform in isolated_transforms.items():
        my_dataset = TransformedImageDataset(
            data_path="/home/ahmet/PycharmProjects/CMPE492/pairwise_adv_dataset",
            processor=processor,
            classes=classes,
            transform_function=transform
        )

        # 2. Wrap it in a DataLoader
        # batch_size=32 means it will process and return 32 images at a time.
        # num_workers=4 uses multiple CPU cores to load images faster.
        dataloader = DataLoader(my_dataset, batch_size=8, shuffle=True, num_workers=4)

        Experiments.analyse_pairwise_corruption(model_nnsight, dataloader, title, F.mse_loss, measure_at_logit="in")

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




