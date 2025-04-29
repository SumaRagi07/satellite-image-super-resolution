import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize
from torchvision.transforms.functional import gaussian_blur
import torch.nn.functional as F
from torchvision import transforms



# Set the paths to the LR and HR folders for bicubic interpolation and Gaussian blur
bicubic_lr_folder = r'\Bicubic_Interpolation\LR'
bicubic_hr_folder = r'\Bicubic_Interpolation\HR'

gaussian_lr_folder = r'\Gaussian_Blur\LR'
gaussian_hr_folder = r'\Gaussian_Blur\HR'

# Set hyperparameters
num_epochs = 10
batch_size = 8
lr = 0.001

# Define the LR-HR dataset class
class LrHrDataset(Dataset):
    def __init__(self, lr_folder, hr_folder, transform=None):
        self.lr_filenames = sorted(os.listdir(lr_folder))
        self.hr_filenames = sorted(os.listdir(hr_folder))
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.transform = transform

    def __len__(self):
        return len(self.lr_filenames)

    def __getitem__(self, index):
        lr_image = Image.open(os.path.join(self.lr_folder, self.lr_filenames[index])).convert("RGB")
        hr_image = Image.open(os.path.join(self.hr_folder, self.hr_filenames[index])).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

class AdaptiveResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out

class AdaptiveSRCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.residual_block = AdaptiveResidualBlock(64, 64)  # Specify input and output channels
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.residual_block(x)  # Apply the adaptive residual block
        x = self.conv3(x)
        return x



# Set the desired transformations for the LR-HR dataset
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Custom collate function
def custom_collate_fn(batch):
    lr_images, hr_images = zip(*batch)
    lr_images = torch.stack(lr_images)
    hr_images = torch.stack(hr_images)
    return lr_images, hr_images




# Create the bicubic interpolation dataset
bicubic_dataset = LrHrDataset(bicubic_lr_folder, bicubic_hr_folder, transform=transform)
bicubic_dataloader = DataLoader(bicubic_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)

# Create the Gaussian blur dataset
gaussian_dataset = LrHrDataset(gaussian_lr_folder, gaussian_hr_folder, transform=transform)
gaussian_dataloader = DataLoader(gaussian_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)

# Instantiate the model
model = AdaptiveSRCNN()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
best_loss = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print("Bicubic Interpolation Training:")
    for i, (lr_images, hr_images) in enumerate(bicubic_dataloader):
        # Clear gradients
        optimizer.zero_grad()

        # Move data to GPU if available
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        # Forward pass
        outputs = model(lr_images)

        # Compute loss
        loss = criterion(outputs, hr_images)

        # Backward pass
        loss.backward()

        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Print mini-batch progress
        print(f"\tBatch [{i+1}/{len(bicubic_dataloader)}], Loss: {loss.item()}")

    print("Gaussian Blur Training:")
    for i, (lr_images, hr_images) in enumerate(gaussian_dataloader):
        # Clear gradients
        optimizer.zero_grad()

        # Move data to GPU if available
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        # Forward pass
        outputs = model(lr_images)

        # Compute loss
        loss = criterion(outputs, hr_images)

        # Backward pass
        loss.backward()

        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Print mini-batch progress
        print(f"\tBatch [{i+1}/{len(gaussian_dataloader)}], Loss: {loss.item()}")

    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    # Save the model if the loss is decreasing
    if best_loss is None or loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), 'Adaptsrcnn1.pth')

# Save the final trained model
torch.save(model.state_dict(), 'adaptsrcnn2.pth')


