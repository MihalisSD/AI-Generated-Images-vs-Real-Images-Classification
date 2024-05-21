import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from PIL import Image

# Custom transformation to convert images to RGBA if necessary
class ConvertToRGBA:
    def __call__(self, img):
        img = img.convert('RGBA')
        return img

# Define image transformations
transform = transforms.Compose([
    #ConvertToRGBA(),  # Convert palette images with transparency to RGBA
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert images to PyTorch tensors and normalize to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize the image
])

# Load dataset
dataset = datasets.ImageFolder(root=r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\data/', transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define a simple ANN
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        kernel_size = 3
        stride = 2
        padding = 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 568)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(568, len(dataset.classes))  # Number of output classes

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')

# Instantiate the model, loss function, and optimizer
model = myCNN().to(device)  # Move the model to the GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Num params: ',sum(p.numel() for p in model.parameters()))
print(summary(model, (3, 128, 128), 32))

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
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
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {running_train_loss / len(train_loader)}, '
          f'Validation Loss: {running_val_loss / len(val_loader)}, '
          f'Accuracy: {100 * correct / total}%')

print('Training complete')
