import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchsummary import summary
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image

# Transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)) 
])

# Load dataset
dataset = datasets.ImageFolder(root=r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\train_val', transform=transform)

# ANN
class myANN(nn.Module):
    def __init__(self, num_classes=len(dataset.classes)):
        super(myANN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
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
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')


num_epochs = 1
batch_size = 32
learning_rate = 0.001
k_folds = 2

kfold = KFold(n_splits=k_folds, shuffle=True)

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
train_loss_list = []
val_loss_list = []
confusion_matrices = []
all_labels = []
all_preds = []

#============
#  Training
#============

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

    print('--------------------------------')
    print(f'FOLD {fold + 1}')
    
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

    model = myANN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    
    fold_train_loss = []
    fold_val_loss = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        fold_train_loss.append(running_train_loss / len(train_loader))

        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        fold_val_loss.append(running_val_loss / len(val_loader))

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {running_train_loss / len(train_loader)}, '
              f'Validation Loss: {running_val_loss / len(val_loader)}, '
              f'Accuracy: {100 * correct / total}%')

    train_loss_list.append(np.mean(fold_train_loss))
    val_loss_list.append(np.mean(fold_val_loss))

    # Evaluation for this fold
    all_fold_labels = []
    all_fold_preds = []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_fold_labels.extend(labels.cpu().numpy())
            all_fold_preds.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_fold_labels, all_fold_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_fold_labels, all_fold_preds, average='weighted')
    
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    confusion_matrices.append(confusion_matrix(all_fold_labels, all_fold_preds))
    all_labels.extend(all_fold_labels)
    all_preds.extend(all_fold_preds)

# Calculate mean and std for metrics
mean_accuracy = np.mean(accuracy_list)
std_accuracy = np.std(accuracy_list)
mean_precision = np.mean(precision_list)
std_precision = np.std(precision_list)
mean_recall = np.mean(recall_list)
std_recall = np.std(recall_list)
mean_f1 = np.mean(f1_list)
std_f1 = np.std(f1_list)
mean_train_loss = np.mean(train_loss_list)
std_train_loss = np.std(train_loss_list)
mean_val_loss = np.mean(val_loss_list)
std_val_loss = np.std(val_loss_list)

print(f'Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}')
print(f'Precision: {mean_precision:.2f} ± {std_precision:.2f}')
print(f'Recall: {mean_recall:.2f} ± {std_recall:.2f}')
print(f'F1 Score: {mean_f1:.2f} ± {std_f1:.2f}')
print(f'Train Loss: {mean_train_loss:.4f} ± {std_train_loss:.4f}')
print(f'Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}')

# Plot confusion matrix
cm = np.sum(confusion_matrices, axis=0)
plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(dataset.classes))
plt.xticks(tick_marks, dataset.classes, rotation=45)
plt.yticks(tick_marks, dataset.classes)

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Save the trained model
model_save_path = (r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\model_myANN.pt')
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
