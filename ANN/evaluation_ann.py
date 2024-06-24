import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from PIL import Image

# my ANN
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

def evaluate_model(model, device, test_loader, dataset_classes):
    model.eval()
    test_correct = 0
    test_total = 0
    test_all_labels = []
    test_all_preds = []
    misclassified_info = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_all_labels.extend(labels.cpu().numpy())
            test_all_preds.extend(predicted.cpu().numpy())

            # Identify misclassified samples and store their info
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified_info.append({
                        'path': test_loader.dataset.samples[(batch_idx * test_loader.batch_size) + i][0],
                        'true_label': labels[i].cpu().numpy(),
                        'predicted_label': predicted[i].cpu().numpy()
                    })

    test_accuracy = accuracy_score(test_all_labels, test_all_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_all_labels, test_all_preds, average='weighted')
    test_cm = confusion_matrix(test_all_labels, test_all_preds)

    return test_accuracy, test_precision, test_recall, test_f1, test_cm, misclassified_info

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_misclassified_images(misclassified_info, dataset_classes, n_images=10):
    plt.figure(figsize=(15, 10))
    for idx, mis_info in enumerate(misclassified_info[:n_images]):
        img_path = mis_info['path']
        image = Image.open(img_path)
        true_label = dataset_classes[mis_info['true_label']]
        predicted_label = dataset_classes[mis_info['predicted_label']]

        plt.subplot(2, n_images // 2, idx + 1)
        plt.imshow(image)
        plt.title(f'True: {true_label}, Pred: {predicted_label}')
        plt.axis('off')
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

my_test_dataset = datasets.ImageFolder(root=r'/content/drive/MyDrive/my_test', transform=transform)
my_test_loader = DataLoader(my_test_dataset, batch_size=64, shuffle=False, num_workers=2)

test_data_dataset = datasets.ImageFolder(root=r'/content/drive/MyDrive/test_data', transform=transform)
test_data_loader = DataLoader(test_data_dataset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Evaluating on device: {device}')

# Load the trained model
model_t_cnn = my_cnn(num_classes=2).to(device)
model_load_path = r'/content/drive/MyDrive/model_cnn.pt'
model_t_cnn.load_state_dict(torch.load(model_load_path, map_location=device))

# Evaluation on my_test set
print("Evaluating on my_test set:")
my_test_accuracy, my_test_precision, my_test_recall, my_test_f1, my_test_cm, my_test_misclassified_info = evaluate_model(model_t_cnn, device, my_test_loader, my_test_dataset.classes)
print(f'My Test Accuracy: {my_test_accuracy:.2f}')
print(f'My Test Precision: {my_test_precision:.2f}')
print(f'My Test Recall: {my_test_recall:.2f}')
print(f'My Test F1 Score: {my_test_f1:.2f}')
plot_confusion_matrix(my_test_cm, my_test_dataset.classes, title='Confusion Matrix (My Test Set)')

# Plot misclassified images
print(f'Misclassified images in my_test set:')
plot_misclassified_images(my_test_misclassified_info, my_test_dataset.classes)

# Evaluation on test_data set
print("Evaluating on test_data set:")
test_data_accuracy, test_data_precision, test_data_recall, test_data_f1, test_data_cm, test_data_misclassified_info = evaluate_model(model_t_cnn, device, test_data_loader, test_data_dataset.classes)
print(f'Test Data Accuracy: {test_data_accuracy:.2f}')
print(f'Test Data Precision: {test_data_precision:.2f}')
print(f'Test Data Recall: {test_data_recall:.2f}')
print(f'Test Data F1 Score: {test_data_f1:.2f}')
plot_confusion_matrix(test_data_cm, test_data_dataset.classes, title='Confusion Matrix (Test Data Set)')

# Plot misclassified images
print(f'Misclassified images in test_data set:')
plot_misclassified_images(test_data_misclassified_info, test_data_dataset.classes)