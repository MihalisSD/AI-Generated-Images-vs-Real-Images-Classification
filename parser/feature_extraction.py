import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 as required by ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize with mean and std
])

# Load the dataset
dataset = datasets.ImageFolder(root=r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\data/', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

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

# Extract features and labels
features, labels = extract_features(dataloader, model)

# Convert to a DataFrame and save to CSV
df_features = pd.DataFrame(features)
df_labels = pd.DataFrame(labels, columns=['label'])
df = pd.concat([df_features, df_labels], axis=1)
df.to_csv(r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\features.csv', index=False)

print("Features and labels saved to features.csv")
