import os
import cv2

def apply_gaussian_blur(input_folder, output_folder, kernel_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)

        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # Save the low-resolution image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, blurred_img)

    print("Gaussian blur applied to all images.")

# Set the input and output folders
input_folder = r'\train_1000img\Gaussian_Blur\HR'
output_folder = r'\train_1000img\Gaussian_Blur\LR'

# Set the kernel size for Gaussian blur
kernel_size = 7

# Apply Gaussian blur to the images in the input folder
apply_gaussian_blur(input_folder, output_folder, kernel_size)
