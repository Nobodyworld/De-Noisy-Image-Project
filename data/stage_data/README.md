# ImageMLDataset
## All You Need To Know

1. **Stage Dir** The `/stage_data/` directory is the main image processing space and is inteneded to be a consolidated directory to allow for edits, transformations, resizeing, and most importantly, allocation to the `/test/`, `/train/` and `/val/` directories for training an ML model. 

2. **Move Images To Training Directories**
Run `stage_move_imgs.py` to move images into `/test`, `/train`, and `/val`. To adjust the ratio of images that are split between these three directories go to lines 30-40 in `stage_move_imgs.py` and adjust the lines below.

```python
    num_images = len(images)
    num_train = int(num_images * 0.80) # Percent of dataset allocated to `/train`
    num_val = int(num_images * 0.12) # Percent of dataset allocated to `/val`
    # The remaining percent is allocated to `/test`. In this case, (.08) or eight percent.
```

3. **Get Images From Training Directories** To get images back to `/data/stage_data` use the script `stage_get_imgs.py`. This <strong>should</strong> be done if you intend to use any of the other scripts in `/data_designer/`

4. **Additional Scripts** The other `.py` files in `/data_designer/` should be used cautiously. They have hard coded variables such as `/path/to/stage_data`, a rename function, and image resizing variables like those pasted below. 

`stage_rename_all.py`
```python
def rename_files(directory, prefix='github_', suffix='_before', extension='.jpg'):
```

`stage_resize.py`
```python
# Set image dimensions
    img_height = 1920
    img_width = 1280
```
**Make sure to adjust these according to your compute needs or when attempting to add new data to the training directory.**


### Note: 
The directory `/play_data` is intended to be pointed at from an external ML directory and was created with the intention to ease inference for an "image input" based nerual network. `/play_data/output` is the intended directory for inferenced or transformed play images.



## Working with Larger File Sizes vs. Image Sizes (width x height)
When working with images in a U-Net architecture in PyTorch, the factor that directly impacts GPU memory consumption is the dimensions of the image (i.e., width × height), not the file size on disk. The file size of an image in formats like JPEG or PNG is not directly correlated to the amount of GPU memory used during training or inference. Instead, the GPU memory usage is determined by the size of the tensors into which these images are loaded and transformed.

Here's a breakdown of how images are processed and how memory is utilized:

1. **Loading Images**: When you load images from disk (e.g., JPG files), they are read into system memory (RAM). The file size might affect loading times and RAM usage, but not GPU memory directly.

2. **Transforming Images**: Before or during training, images are often transformed (e.g., resized, normalized) to ensure that they are of a consistent size and format for the neural network. These transformations convert images from their compressed file formats into raw pixel data stored in tensors.

3. **Transferring to GPU**: When tensors are transferred to the GPU for computation, the amount of GPU memory they consume depends on the dimensions of the tensor (which correlate to the dimensions of the transformed image) and the data type of the tensor elements (e.g., float32, float16). For instance, a color image tensor of size `[3, 960, 640]` (3 channels, 960x640 pixels) with `float32` values would consume `3 * 960 * 640 * 4` bytes of GPU memory, irrespective of the original file size of the image.

4. **Batch Processing**: During training, images are usually grouped into batches. The batch size further multiplies the memory requirement. A batch of 32 images of the size mentioned above would consume 32 times more memory than a single image tensor.

### Handling High-Quality Images

If you want to include high-quality images in your `/data` directory and have the flexibility to downscale them if needed, you can indeed use PyTorch's `torchvision.transforms` to dynamically resize images during the dataset loading phase. This approach allows you to store high-quality images on disk while controlling their impact on GPU memory by adjusting their size before training or inference.

Example using `transforms`:

```python
from torchvision import transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
    transforms.Resize((desired_height, desired_width)),  # Resize to the desired size
    transforms.ToTensor(),
    # Add other transformations here
])

dataset = ImageFolder(root='./data/train', transform=transform)
```

This method is memory efficient because the transformations are applied on-the-fly during data loading, meaning only the transformed images occupy GPU memory, not the original high-resolution images.

By strategically managing image dimensions with transformations, you can optimize GPU memory usage without being constrained by the file sizes of your high-quality images.