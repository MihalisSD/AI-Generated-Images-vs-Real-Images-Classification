import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import os
import time

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 as required by ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std
])

# Load the original dataset
original_dataset = datasets.ImageFolder(root=r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\train_val', transform=transform)

# Load the augmented dataset
augmented_dataset = datasets.ImageFolder(root=r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\augmentation_train_val', transform=transform)

# Combine the original and augmented datasets
combined_dataset = ConcatDataset([original_dataset, augmented_dataset])

# Create DataLoader for the combined dataset
dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

# Load the test dataset
test_dataset = datasets.ImageFolder(root=r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\test', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Remove the last fully connected layer
model = nn.Sequential(*list(model.children())[:-1])

def extract_features(dataloader, model):
    features = []
    labels = []
    with torch.no_grad():
        for images, label_batch in dataloader:
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten the outputs
            features.append(outputs.cpu().numpy())
            labels.append(label_batch.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# Start the timer for training set
start_time_train = time.time()

# Extract features and labels for training set
features_train, labels_train = extract_features(dataloader, model)

# End the timer for training set
end_time_train = time.time()
elapsed_time_train = end_time_train - start_time_train

# Convert elapsed time to hours, minutes, and seconds for training set
hours_train, rem_train = divmod(elapsed_time_train, 3600)
minutes_train, seconds_train = divmod(rem_train, 60)
print(f"Training set feature extraction completed in {int(hours_train)}h {int(minutes_train)}m {int(seconds_train)}s")

# Verify class distribution for training set
unique_train, counts_train = np.unique(labels_train, return_counts=True)
print(f"Training set class distribution: {dict(zip(unique_train, counts_train))}")

# Convert training set features and labels to DataFrame and save to CSV
df_features_train = pd.DataFrame(features_train)
df_labels_train = pd.DataFrame(labels_train, columns=['label'])
df_train = pd.concat([df_features_train, df_labels_train], axis=1)
df_train.to_csv(r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\features.csv', index=False)

# Start the timer for test set
start_time_test = time.time()

# Extract features and labels for test set
features_test, labels_test = extract_features(test_dataloader, model)

# End the timer for test set
end_time_test = time.time()
elapsed_time_test = end_time_test - start_time_test

# Convert elapsed time to hours, minutes, and seconds for test set
hours_test, rem_test = divmod(elapsed_time_test, 3600)
minutes_test, seconds_test = divmod(rem_test, 60)
print(f"Test set feature extraction completed in {int(hours_test)}h {int(minutes_test)}m {int(seconds_test)}s")

# Verify class distribution for test set
unique_test, counts_test = np.unique(labels_test, return_counts=True)
print(f"Test set class distribution: {dict(zip(unique_test, counts_test))}")

# Convert test set features and labels to DataFrame and save to CSV
df_features_test = pd.DataFrame(features_test)
df_labels_test = pd.DataFrame(labels_test, columns=['label'])
df_test = pd.concat([df_features_test, df_labels_test], axis=1)
df_test.to_csv(r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\test_features.csv', index=False)

print("Features and labels saved to features.csv and test_features.csv")
