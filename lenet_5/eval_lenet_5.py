import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# LeNet-5 Model Definition
class lenet(nn.Module):
    def __init__(self, num_classes):
        super(lenet, self).__init__()
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

def main():
    # Transformations
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(root=r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)  # Set num_workers=0 for debugging

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating on device: {device}')

    # Load the trained model
    model = lenet(num_classes=len(test_dataset.classes)).to(device)
    print(len(test_dataset.classes))
    model_load_path = r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\model_lenet_5.pt'
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.eval()
    print(f'Model loaded from {model_load_path}')

    #==========================
    #  Evaluation on Test Set
    #==========================

    test_correct = 0
    test_total = 0
    test_all_labels = []
    test_all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_all_labels.extend(labels.cpu().numpy())
            test_all_preds.extend(predicted.cpu().numpy())

    # Calculate metrics for the test set
    test_accuracy = accuracy_score(test_all_labels, test_all_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_all_labels, test_all_preds, average='weighted')
    test_cm = confusion_matrix(test_all_labels, test_all_preds)

    print(f'Test Accuracy: {test_accuracy:.2f}')
    print(f'Test Precision: {test_precision:.2f}')
    print(f'Test Recall: {test_recall:.2f}')
    print(f'Test F1 Score: {test_f1:.2f}')

    # Plot confusion matrix for the test set
    plt.figure(figsize=(10, 7))
    plt.imshow(test_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Set Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(test_dataset.classes))
    plt.xticks(tick_marks, test_dataset.classes, rotation=45)
    plt.yticks(tick_marks, test_dataset.classes)

    thresh = test_cm.max() / 2.
    for i, j in product(range(test_cm.shape[0]), range(test_cm.shape[1])):
        plt.text(j, i, format(test_cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if test_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    main()
