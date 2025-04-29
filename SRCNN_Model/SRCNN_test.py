import os
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image

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

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = SRCNN()
model.load_state_dict(torch.load("srcnn_model.pth"))
model = model.to(device)
model.eval()

# Paths
test_folder = r"C:\Users\DELL\Desktop\ADRIN\srcnn_imp\new_data\test"
output_folder = r"C:\Users\DELL\Desktop\ADRIN\srcnn_imp\test_results"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of image files in the test folder
image_files = [file for file in os.listdir(test_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image in the test folder
for image_file in image_files:
    # Load the test image
    test_image = Image.open(os.path.join(test_folder, image_file)).convert("RGB")

    # Preprocess the test image
    transform = ToTensor()
    input_image = transform(test_image).unsqueeze(0).to(device)

    # Perform super-resolution
    with torch.no_grad():
        output_image = model(input_image)

    # Convert the output tensor to an image
    output_image = output_image.squeeze(0).cpu().clamp(0, 1).numpy()
    output_image = (output_image * 255).astype("uint8")
    output_image = Image.fromarray(output_image.transpose(1, 2, 0))

    # Save the output image
    output_path = os.path.join(output_folder, image_file)
    output_image.save(output_path)

print("Testing completed!")

