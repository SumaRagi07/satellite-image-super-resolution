import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os

# Define the SRCNN model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Define the training function
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    total_samples = len(train_loader.dataset)
    print(f"Training on {total_samples} samples...")
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            print(f"Input shape: {inputs.shape}")  # Add this line

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            for idx in range(len(inputs)):
                processed_samples = batch_idx * train_loader.batch_size + (idx + 1)
                progress = processed_samples / total_samples * 100
                print(f"Epoch [{epoch + 1}/{epochs}], Image [{processed_samples}/{total_samples}], Loss: {loss.item():.4f}, Progress: {progress:.2f}%")

        print(f"Epoch {epoch + 1} Loss: {running_loss / len(train_loader)}")

    print("Training finished!")


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the hyperparameters
lr = 0.001
epochs = 10
batch_size = 1

# Define a custom dataset class
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print(f"Processing image: {img_path}")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image


# Load the training dataset
dataset = ImageDataset(r"C:\Users\DELL\Desktop\ADRIN\srcnn_imp\new_data\train", transform=ToTensor())
print("Dataset length:", len(dataset))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the SRCNN model instance and move it to the device
model = SRCNN().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Training started...")
# Train the SRCNN model
train(model, train_loader, criterion, optimizer, epochs)
print("Training finished!")

# Save the trained model
torch.save(model.state_dict(), "srcnn_model.pth")


