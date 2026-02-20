import os
import torch
from PIL.ImageChops import offset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSetups:
    @staticmethod
    def generate_pairwise_dataset(data_path, processor, true_label, distorted_label):
        path_clean = os.path.join(data_path, true_label, true_label)
        path_corrupted = os.path.join(data_path, true_label, distorted_label)
        files_clean = os.listdir(path_clean)
        files_corrupted = os.listdir(path_corrupted)
        dpoints_clean = []
        dpoints_adversarial = []
        for file in files_clean:
            dpoint_clean = Image.open(os.path.join(path_clean, file)).convert("RGB")
            dpoints_clean.append(dpoint_clean)
            dpoint_adversarial = Image.open(os.path.join(path_corrupted, file)).convert("RGB")
            dpoints_adversarial.append(dpoint_adversarial)

        input_adversarial = processor(images=dpoints_adversarial, return_tensors="pt")["pixel_values"].to(device)
        input_clean = processor(images=dpoints_clean, return_tensors="pt")["pixel_values"].to(device)

        return input_clean, input_adversarial

    @staticmethod
    def generate_transformed_dataset(data_path, processor, classes, transform_function):
        dpoints_clean = []
        dpoints_transformed = []

        # Iterate through the provided list of class directories
        for class_name in classes:
            class_path = os.path.join(data_path, class_name, class_name)

            # Skip if the directory doesn't exist
            if not os.path.isdir(class_path):
                continue

            files = os.listdir(class_path)

            for file in files:
                # Ensure we are only reading PNG images
                if file.lower().endswith('.png'):
                    img_path = os.path.join(class_path, file)

                    # 1. Read clean image
                    dpoint_clean = Image.open(img_path).convert("RGB")
                    dpoints_clean.append(dpoint_clean)

                    # 2. Apply the provided transformation function
                    dpoint_transformed = transform_function(dpoint_clean)
                    dpoints_transformed.append(dpoint_transformed)

        input_clean = processor(images=dpoints_clean, return_tensors="pt")["pixel_values"].to(device)
        input_transformed = processor(images=dpoints_transformed, return_tensors="pt")["pixel_values"].to(device)

        return input_clean, input_transformed

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