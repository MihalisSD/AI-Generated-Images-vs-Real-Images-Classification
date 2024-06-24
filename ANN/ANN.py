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
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),  # Convert images to PyTorch tensors and normalize to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize the image
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class my_cnn(nn.Module):
    def __init__(self, num_classes=2):
        super(my_cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3) #256*256*32 same ----> 7x7 conv
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #128*128*32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) #same 128*128*64 -------> 5x5 conv
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2) #same 128*128*128
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 64*64*128

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) #64*64*256 --------> 3x3 conv
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256, 362, kernel_size=3, stride=1, padding=1) #64*64*362
        self.bn5 = nn.BatchNorm2d(362)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # 32*32*362

        self.conv6 = nn.Conv2d(362, 512, kernel_size=1, stride=1, padding=0) #32*32*512 ------> 1x1 conv 
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(512, 724, kernel_size=1, stride=1, padding=0) #32*32*724
        self.bn7 = nn.BatchNorm2d(724)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(724, 1024, kernel_size=1, stride=1, padding=0) #32*32*1024
        self.bn8 = nn.BatchNorm2d(1024)
        self.relu8 = nn.ReLU()
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2) # 16*16*1024

        self.fc9 = nn.Linear(16*16*1024, 724)
        self.relu9 = nn.ReLU()
        self.dropout9 = nn.Dropout(0.5)

        self.fc10 = nn.Linear(724, 516)
        self.relu10 = nn.ReLU()
        self.dropout10 = nn.Dropout(0.5)

        self.fc11 = nn.Linear(516, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.pool8(x)

        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc9(x)
        x = self.relu9(x)
        x = self.dropout9(x)

        x = self.fc10(x)
        x = self.relu10(x)
        x = self.dropout10(x)

        x = self.fc11(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')

model_cnn = my_cnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

print('Num params: ', sum(p.numel() for p in model_cnn.parameters()))
print(summary(model_cnn, (3, 256, 256), 64))

# Training loop
num_epochs = 50

train_losses = []
val_losses = []

start_time_train = time.time()
for epoch in range(num_epochs):
    model_cnn.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the GPU

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model_cnn(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model

        running_train_loss += loss.item()

    # Validation step
    model_cnn.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_cnn(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())  # Assuming binary classification

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
model_save_path = r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\model_myANN.pt'
torch.save(model_cnn.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
