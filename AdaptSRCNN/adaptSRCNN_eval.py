import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
from scipy import signal
from skimage.color import rgb2gray


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

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the SRCNN model
model = SwinIRModel().to(device)

# Load the trained model weights
model.load_state_dict(torch.load("swinir_model2.pth"))

# Set the model to evaluation mode
model.eval()

# Paths
test_folder = r"\test_lr"
output_folder = r"\Outputs"

# Evaluate on test images
psnr_values = []
ssim_values = []

for image_file in os.listdir(test_folder):
    # Load the test image
    image_path = os.path.join(test_folder, image_file)
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    transform = ToTensor()
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        sr_image = model(input_image)

    # Convert the output image tensor to a PIL Image
    sr_image = sr_image.squeeze(0).cpu().clamp(0, 1).numpy()
    sr_image = np.transpose(sr_image, (1, 2, 0))
    sr_image = (sr_image * 255).astype(np.uint8)
    sr_image = Image.fromarray(sr_image)

    # Save the super-resolved image
    output_path = os.path.join(output_folder, image_file)
    sr_image.save(output_path)

    # Calculate PSNR
    gt_image = np.array(image)
    sr_image = np.array(sr_image)
    psnr = peak_signal_noise_ratio(gt_image, sr_image)
    psnr_values.append(psnr)

    # Convert images to grayscale
    gt_gray = rgb2gray(gt_image)
    sr_gray = rgb2gray(sr_image)

    # Calculate SSIM
    window = np.ones((11, 11))
    mu1 = signal.convolve2d(gt_gray, window, mode='valid')
    mu2 = signal.convolve2d(sr_gray, window, mode='valid')
    sigma1_sq = signal.convolve2d(gt_gray * gt_gray, window, mode='valid') - mu1 * mu1
    sigma2_sq = signal.convolve2d(sr_gray * sr_gray, window, mode='valid') - mu2 * mu2
    sigma12 = signal.convolve2d(gt_gray * sr_gray, window, mode='valid') - mu1 * mu2

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim = np.mean(ssim_map)

    ssim_values.append(ssim)

    print(f"Image: {image_file}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

# Calculate the average PSNR and SSIM values
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")

print("Evaluation completed.")
