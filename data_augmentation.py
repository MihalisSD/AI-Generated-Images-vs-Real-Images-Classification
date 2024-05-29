import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch

def augment_and_save(image_tensor, transform, save_path, class_name, original_extension, augmentation_type, counter):
    image = transforms.ToPILImage()(image_tensor)
    augmented_image = transform(image)
    augmented_image.save(os.path.join(save_path, f"{class_name}_img_{augmentation_type}_{counter}{original_extension}"))

def data_augmentation(input_dir, output_dir):  #Augmentations
    vertical_flip = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0), #Mirror
    ])
    
    rotation = transforms.Compose([
        transforms.RandomRotation(15),  # Rotate by ±15 degrees
    ])
    
    random_crop = transforms.Compose([
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),  # Crop
    ])
    
    gaussian_noise = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),  # Add Gaussian noise
        transforms.ToPILImage()
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root=input_dir, transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Class name mapping
    class_name_mapping = {
        0: "ai_aug",
        1: "real_aug"}
    
    for idx, (inputs, labels) in enumerate(dataloader):
        label = labels.item()
        class_name = class_name_mapping[label]
        class_dir = os.path.join(output_dir, class_name)
        
        # Ensure class subdirectory exists
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        original_file_path = dataset.samples[idx][0]
        original_extension = os.path.splitext(original_file_path)[1]

        # Apply
        augment_and_save(inputs[0], vertical_flip, class_dir, class_name, original_extension, "flip", idx + 1)
        augment_and_save(inputs[0], rotation, class_dir, class_name, original_extension, "rotation", idx + 1)
        augment_and_save(inputs[0], random_crop, class_dir, class_name, original_extension, "crop", idx + 1)
        augment_and_save(inputs[0], gaussian_noise, class_dir, class_name, original_extension, "noise", idx + 1)

# Paths
input_directory = r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\train_val'
output_directory = r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\augmentation_train_val'

# Apply data augmentation
data_augmentation(input_directory, output_directory)
