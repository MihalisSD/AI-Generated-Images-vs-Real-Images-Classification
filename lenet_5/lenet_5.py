import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchsummary import summary
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

generator1 = torch.Generator().manual_seed(42)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images
    transforms.ToTensor(),  # Convert images to PyTorch tensors and normalize to [0, 1]
    transforms.Normalize((0.5,), (0.5,)) # normalize the image
])

# Load original dataset
original_dataset = datasets.ImageFolder(root=r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\train_val', transform=transform)

# Load augmented dataset
augmented_dataset = datasets.ImageFolder(root=r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\augmentation_train_val', transform=transform)

# Concatenate datasets
combined_dataset = ConcatDataset([original_dataset, augmented_dataset])

# Split combined dataset into training and validation sets
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size], generator=generator1)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Lenet-5
class lenet(nn.Module):
    def __init__(self, num_classes=2):
        super(lenet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(16 * 5 * 5, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')

# Instantiate the model, loss function, and optimizer
model = lenet().to(device)  # Move the model to the GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Num params: ', sum(p.numel() for p in model.parameters()))
print(summary(model, (3, 32, 32), 32))

# Training loop
num_epochs = 15

train_losses = []
val_losses = []

start_time_train = time.time()
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the GPU

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model

        running_train_loss += loss.item()

    # Validation step
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())

    accuracy = 100 * correct / total
    train_losses.append(running_train_loss / len(train_loader))
    val_losses.append(running_val_loss / len(val_loader))

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_losses[-1]}, '
          f'Validation Loss: {val_losses[-1]}, '
          f'Accuracy: {accuracy}%')

# Time
end_time_train = time.time()
elapsed_time_train = end_time_train - start_time_train
hours_train, rem_train = divmod(elapsed_time_train, 3600)
minutes_train, seconds_train = divmod(rem_train, 60)
print(f"Training completed in :  {int(hours_train)}h {int(minutes_train)}m {int(seconds_train)}s")

# Plot training and validation loss
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Final results
report = classification_report(all_labels, all_preds, target_names=original_dataset.classes, output_dict=True)
cm = confusion_matrix(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
fpr, tpr, _ = roc_curve(all_labels, all_probs)

print(f'Final Results: '
      f'Accuracy: {accuracy}%, '
      f'Precision: {report["weighted avg"]["precision"]}, '
      f'Recall: {report["weighted avg"]["recall"]}, '
      f'F1 Score: {report["weighted avg"]["f1-score"]}, '
      f'AUC: {auc}')

# Print confusion matrix
print('Confusion Matrix:')
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=original_dataset.classes, yticklabels=original_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Save the trained model
model_save_path = r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\model_lenet_5.pt'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
