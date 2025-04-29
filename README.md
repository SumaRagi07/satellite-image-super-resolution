# Super-Resolution Techniques for Satellite Images

This repository contains implementations of three super-resolution models developed during an internship project at ADRIN (Advanced Data Research Institute), ISRO:
- **SRCNN (Super-Resolution Convolutional Neural Network)**
- **LAPSRN (Laplacian Pyramid Super-Resolution Network)**
- **AdaptSRCNN (An adapted lightweight SRCNN model)**

The goal was to explore deep learning-based super-resolution techniques for enhancing the quality of satellite imagery.

## Project Structure

```
📂 SuperResolution-Satellite-Images/
 ├── SRCNN/
 │    ├── srcnn_main.py           # Model training script
 │    ├── srcnn_eval.py           # Evaluation script (PSNR, SSIM)
 │    ├── srcnn_test.py           # Testing and inference script
 │    ├── convert_to_jpg.py       # Script to preprocess dataset
 │
 ├── LAPSRN/
 │    ├── lapsrn_training.ipynb   # Full LAPSRN model training notebook
 │
 ├── AdaptSRCNN/
 │    ├── adaptsrcnn.py           # Model training script
 │    ├── adapt_eval.py           # Evaluation script (PSNR, SSIM)
 │    ├── adapt_test.py           # Inference on test images
 │    ├── lr_bicubic_interpolation.py # Script to generate bicubic LR images
 │    ├── lr_gaussian.py          # Script to generate gaussian blurred LR images
 │
 ├── README.md
```

>  **Note**: Model weight files like `.pth` (e.g., `srcnn_model.pth`, `adaptsrcnn.pth`) are **not uploaded** here to keep the repo lightweight.

## Models Overview

- **SRCNN**  
  A simple three-layer convolutional model that directly learns an end-to-end mapping between low-resolution and high-resolution images.

- **LAPSRN**  
  A deeper network based on the Laplacian pyramid framework, progressively predicting residuals and upscaling in stages (2× and 4×).

- **AdaptSRCNN**  
  A modified lightweight network initially intended to be a SwinIR variant but later adapted into a simpler CNN-based architecture for faster convergence.

## Dataset

- **Massachusetts Roads Dataset** from Kaggle.
- Preprocessing included:
  - Format conversion (TIFF → JPEG)
  - Downscaling using **bicubic interpolation** and **Gaussian blur** to simulate low-resolution images.

## Training Details

- Framework: **PyTorch**
- Loss Functions: **MSELoss** (SRCNN, AdaptSRCNN), **Charbonnier Loss** (LAPSRN)
- Evaluation Metrics: **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index)


## Understanding PSNR and SSIM

- **PSNR (Peak Signal-to-Noise Ratio)** is a common metric to measure the quality of a reconstructed image compared to the original.  
  - Higher PSNR generally indicates better reconstruction quality.
  - In super-resolution tasks:
    - **PSNR > 30 dB** is considered *very good*.
    - **25–30 dB** is *moderate but acceptable* for challenging datasets.
    - **< 25 dB** often indicates visible artifacts or loss of fine details.

- **SSIM (Structural Similarity Index)** measures the perceived quality of the image based on structural information like luminance, contrast, and texture.
  - SSIM values range from **0 to 1**, where **closer to 1 means higher similarity**.
  - **SSIM > 0.9** is generally considered very good for super-resolution tasks.

> In this project, PSNR and SSIM improvements across models reflected better reconstruction of satellite imagery details, especially with the AdaptSRCNN variant.

## Results Summary

| Model         | Avg. PSNR | Avg. SSIM |
|---------------|-----------|-----------|
| SRCNN         | 27.76 dB  | ~1.013    |
| LAPSRN        | 20.62 dB  | —         |
| AdaptSRCNN    | 34.05 dB  | ~1.010    |

> AdaptSRCNN showed comparatively better PSNR results for the limited dataset used.

## Notes
- The **AdaptSRCNN** model was originally an attempt to experiment with SwinIR-inspired ideas but ended up as a modified convolutional design due to hardware and time constraints.
- This repository is intended for academic archival purposes and learning.  
  Future re-implementations and improvements are welcome!
