import os
import torch
from PIL.ImageChops import offset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TargetCorruptedImageDataset(Dataset):
    def __init__(self, data_path, processor, true_label, distorted_label, offset = 0):
        self.offset = offset
        self.image_paths_clean = []
        self.image_paths_adversarial = []
        self.processor = processor

        files = os.listdir(os.path.join(data_path, true_label, true_label))

        for file in files:
            self.image_paths_clean.append(os.path.join(data_path, true_label, true_label, file))
            self.image_paths_adversarial.append(os.path.join(data_path, true_label, distorted_label, file))
        self.n_samples = len(self.image_paths_clean)

    def __len__(self):
        # Returns the total number of images
        return self.n_samples

    def __getitem__(self, idx):
        img_path_clean = self.image_paths_clean[(idx + self.offset) % len(self.image_paths_clean)]
        img_path_adv = self.image_paths_adversarial[(idx + self.offset) % len(self.image_paths_adversarial)]

        dpoint_clean = Image.open(img_path_clean).convert("RGB")
        dpoint_adversarial = Image.open(img_path_adv).convert("RGB")

        input_clean = self.processor(images=dpoint_clean, return_tensors="pt")["pixel_values"].squeeze(0)
        input_transformed = self.processor(images=dpoint_adversarial, return_tensors="pt")["pixel_values"].squeeze(0)

        return input_clean, input_transformed


class TransformedImageDataset(Dataset):
    def __init__(self, data_path, processor, classes, transform_function):
        self.processor = processor
        self.transform_function = transform_function
        self.image_paths = []

        # 1. Store ONLY the file paths in memory, not the actual images
        for class_name in classes:
            class_path = os.path.join(data_path, class_name, class_name)

            if not os.path.isdir(class_path):
                continue

            for file in os.listdir(class_path):
                if file.lower().endswith('.png'):
                    self.image_paths.append(os.path.join(class_path, file))

    def __len__(self):
        # Returns the total number of images
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        dpoint_clean = Image.open(img_path).convert("RGB")

        dpoint_transformed = self.transform_function(dpoint_clean)

        input_clean = self.processor(images=dpoint_clean, return_tensors="pt")["pixel_values"].squeeze(0)
        input_transformed = self.processor(images=dpoint_transformed, return_tensors="pt")["pixel_values"].squeeze(0)

        return input_clean, input_transformed

class SingleClassCleanDataset(Dataset):
    def __init__(self, data_path, processor, class_name):
        self.processor = processor
        self.image_paths = []

        class_path = os.path.join(data_path, class_name, class_name)

        if not os.path.isdir(class_path):
            return

        for file in os.listdir(class_path):
            if file.lower().endswith('.png'):
                self.image_paths.append(os.path.join(class_path, file))

    def __len__(self):
        return 16 #len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        dpoint_clean = Image.open(img_path).convert("RGB")

        input_clean = self.processor(images=dpoint_clean, return_tensors="pt")["pixel_values"].squeeze(0)

        return input_clean