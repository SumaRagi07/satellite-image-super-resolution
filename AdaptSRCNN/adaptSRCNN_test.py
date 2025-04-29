import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn as nn

# Define the paths for the test images folder and the output folder
test_folder = r'\test_lr'
output_folder = r'\Outputs'

class SwinIRModel(nn.Module):
    def __init__(self):
        super(SwinIRModel, self).__init__()
        self.in_channels = 3
        self.out_channels = 3
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x
    
# Create an instance of the SwinIR model
model = SwinIRModel()

# Load the saved model state dictionary
state_dict = torch.load('swinir_model_final2.pth')

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Iterate over the images in the test folder
for filename in os.listdir(test_folder):
    image_path = os.path.join(test_folder, filename)
    
    # Load the image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess the image
    lr_image = ToTensor()(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        sr_image = model(lr_image)
    
    # Convert the output tensor to an image
    sr_image = sr_image.squeeze(0).clamp(0, 1).numpy()
    sr_image = (sr_image * 255.0).round().astype('uint8')
    sr_image = Image.fromarray(sr_image.transpose(1, 2, 0), 'RGB')
    
    # Save the resulting enhanced image
    output_path = os.path.join(output_folder, filename)
    sr_image.save(output_path)

print("Inference completed. Enhanced images saved in the output folder.")












