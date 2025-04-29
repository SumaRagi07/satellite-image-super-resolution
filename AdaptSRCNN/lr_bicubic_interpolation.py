import os
import cv2

def apply_bicubic_interpolation(input_folder, output_folder, scale_factor):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of image files in the input folder
    image_files = os.listdir(input_folder)
    
    for filename in image_files:
        # Read the image
        img = cv2.imread(os.path.join(input_folder, filename))
        
        # Perform bicubic interpolation
        img_low_res = cv2.resize(img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Save the low-resolution image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img_low_res)
        
        print(f"Processed {filename} -> Saved as {output_path}")

# Example usage
input_folder = r"\train_1000img\Bicubic_Interpolation\HR"
output_folder = r"\train_1000img\Bicubic_Interpolation\LR"
scale_factor = 5  # Adjust the scale factor as per your requirement

apply_bicubic_interpolation(input_folder, output_folder, scale_factor)


