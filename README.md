# De-Noisy Image Project

This repository now ships with a **fully self-contained** implementation that avoids heavyweight third-party dependencies.  A
tiny Torch-compatible API (`torch/`) and a JSON-backed image library (`PIL/`) keep the project runnable in restricted
environments where installing packages such as PyTorch or Pillow is not possible.  The simplified U-Net still mirrors the
original public API which allows the existing utility scripts to run unchanged.

### Project Overview

Utilizing a U-Net architecture, this project aims to denoise images effectively.  The reference implementation focuses on a
lightweight demonstration model that can run entirely on CPU and operates on small (8x6) sample images stored as JSON files.

### Training Configuration

To adapt to different compute capabilities, the image sizes, number of layers, and several key training parameters can be adjusted. Below is an example configuration:

```json
{
  "training": {
    "epochs": 36,
    "batch_size": 2,
    "accumulation_steps": 1,
    "num_workers": 4,
    "pin_memory": true,
    "shuffle": true,
    "early_stopping": true,
    "early_stopping_patience": 8,
    "step_decrease_interval": 8,
    "img_height": 6,
    "img_width": 8
  }
}
```
## Implementation Notes

### Safe Dimensions
- **Resizing**: Utilize `stage_resize.py` for image resizing. Consider the `odd_pixel_count.png` to determine appropriate starting dimensions, ensuring compatibility with the network's pooling layers.

### Batching
- The batch size reflects how many images are processed together through the network. A smaller batch size does not limit the total number of images seen per epoch but indicates that images are fed in smaller groups. For example, with 20 images and a batch size of 2, there would be 10 batches per epoch.

### Pooling Layers
- With image dimensions of 640x960, up to 7 pooling layers can be applied. This number is derived from halving each dimension until reaching a minimal size that remains a whole number. Adjust the number of pooling layers based on your specific task and network architecture.

### Tensor Mismatch and Ceil Mode
- **Ceil Mode**: Opting for `ceil` instead of `floor` in pooling layers can influence the output shape, potentially leading to tensor mismatches. This option must be carefully managed to ensure compatibility between layers.

### Directory Structure

1. **`./data/`**: Contains subdirectories for different stages of data processing and storage.
2. **`./model_parts/` & `./config/`**: Holds model parameters and configuration settings.
3. **Scripts**: Including `main.py` for training and architecture maintenance, and `test_play_data.py` for inference testing.

### Running the Project

- **Training**: Execute `main.py` with images in the `train`, `val`, and `test` folders.
- **Inference**: Place JSON-formatted images (``*.json``) in `./data/simple/play_data/` and run `test_play_data.py` for denoised
  outputs.  The helper scripts understand the JSON backed images shipped in `data/simple` out of the box.
- **Checkpoints**: The lightweight U-Net ships with a JSON checkpoint (`models/other_model/other_model.json`) so the repository
  remains text-only and can be shared in binary-restricted environments.  Utilities in `utils/checkpointing.py` transparently
  load either the stub JSON files or a conventional PyTorch ``.pth`` file when available.

### Important Considerations

- Ensure compatibility with your hardware, especially GPU capabilities, by running `./tests/test_gpu.py`.
- Backup data before executing scripts that modify or move files to prevent data loss or corruption.
- Adjust parameters in the model or configuration files to tailor the project to your dataset and requirements.

### Model Architecture

- **Encoder**: Convolutional layers with downsampling for feature extraction.
- **Middle Section**: Bridges encoder and decoder, processing feature maps further.
- **Decoder**: Upsamples feature maps, reduces channels, and uses skip connections for feature concatenation.
- **Output**: Final layer mapping feature maps to the target output channels.

### Modifications & Key Features

- Designed for flexibility across different computing environments.
- Supports adjustments in image dimensions and model layers to match computational resources.
- Includes detailed documentation on the model's layers and operational flow.