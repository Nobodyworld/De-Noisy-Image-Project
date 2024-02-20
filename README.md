# De-Noisy-Image-Project Small

This repository contains scripts and directories for the Small "De-Noisy Image Project" using a U-Net architecture. (See "xtra-small" for less needed compute.) The main components are the `resized_imgs` folder for storing and preprocessing images before training, `test_these` for extra inference, and the `model.py` script for training and architecture. The `model.py` will run without an NVIDIA GPU but you should consider reducing the image sizes, the number of layers, or these varaibles mentioned further below. (See lines 190-200 of `model.py`)

  """
    # Set batch size, image dimensions
    batch_size = 8
    img_height = 960
    img_width = 640
    epochs = 48
  """

Note: As you can see above we have reduced the image height and width significantly from the large repo for this project to account for less GPU or CPU compute power. Do not forget to run the `ztest_gpu.py` file to verify you have a connected cuda device.

Batching Note: Having a batch size of two does not mean that my model only sees two images per 'epoch' but rather that my model is fed two images at a time, of all images in the dataset, until it finishes, which completes one epoch. So if I have 20 images I will then have 10 batches per epoch.

For image dimensions of 640 in width and 960 in height, you can apply a maximum of 7 pooling layers. This number is calculated based on how many times each dimension can be divided by 2 (halved) until it reaches a minimum size while remaining a whole number. However, in practice, you may not need or want to use the maximum number of pooling layers, as each pooling layer reduces the spatial resolution of your feature maps. The actual number to use would depend on the specifics of your task and the architecture of your neural network.

"ceil_mode plici(bool) – when True, will use ceil instead of floor to compute the output shape." https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

![image](https://github.com/Nobodyworld/De-Noisy-Image-Project-Small/assets/127373451/ae4f539f-41d8-4ecf-9d2a-f82dfb5b0682)

Tensor mismatch
torch.Size([6, 112, 14, 10])
torch.Size([6, 112, 15, 10])

## Directory Structure

1. **`resized_imgs`**: This folder is used to store images prior to training a model so that you can keep all 'before' and 'after' images in a consolidated location and perform edits should you choose. (I highly recommend not cropping or resizing otherwise you risk skewing the before and after pairs.) After images in this directory are manipulated use 'zmoveimgs.py' to move them proprtionally to `train`, `test`, and `val` folders using the customizable logic below.    
  
  """  
    num_images = len(images)
    num_train = int(num_images * 0.80)
    num_val = int(num_images * 0.12)

  """

Note: If not images are not in train or val an else statement adds them to '/test' within the 'zmoveimgs.py' script. So .08, or 8 percent of the images were allocated to '/test'.

Similarly, if you wanted to get the images back int the '/resized_imgs' folder all you need to do is run the 'zgetimgs.py' script.

Please refer to the 'resized_imgs\readme.txt' file for more info.

2. **`test_these`**: Place your images for additional inference here. Images should be named with a `_before` suffix (e.g., `image_before.jpg`). These images can be processed using the `zrename_test_before.py` script.

3. **`test_these/output`**: This directory will contain the output of the inference process. Images from the `test_these` folder are processed and their denoised versions are saved here.


## Key Scripts

1. **`model.py`**: This script is used for both training the U-Net model and maintaining the architecture referenced during inference.

### Training the Model
Run `model.py` to train the U-Net model. The script will use images from the `train`, `val`, and `test` folders.


2. **`test_these.py`**: Script for manual inference after a model has been trainined.

### Running Manual Inference After Training
1. Place your images for inference in the `test_these` folder, or use the images already there. Ensure additions are named with a `_before` suffix by using the `zrename_test_before.py` script. (Ex: Image1_before.jpg)
2. Run `test_these.py`. It will process the images and save the denoised outputs to the `test_these/output` folder.


## Note
- Ensure that the Python environment has all the necessary libraries installed (PyTorch, torchvision, PIL, etc.).
- Adjust model parameters in `model.py` as needed to suit your dataset and training requirements.
- Always backup your data before running scripts that modify or move files.

---

## Model.py Layers Explained

### Encoder Section
The encoder part of the U-Net architecture consists of several convolutional layers. Each `enc_conv` layer includes two convolutional layers with batch normalization and ReLU activation. The `pool` layers are used for downsampling the feature maps, reducing their dimensions by half.

- `enc_conv1` to `enc_conv8`: These are sequential blocks, each containing two convolutional layers. Each convolution is followed by batch normalization and ReLU activation. These blocks progressively increase the number of channels while capturing complex features from the input images.
- `pool1` to `pool7`: MaxPooling layers used for downsampling. They reduce the spatial dimensions of the feature maps by half, which helps the network to learn increasingly abstract features.
- `res_enc1` to `res_enc7`: These are residual connections in each encoding stage. They help in alleviating the vanishing gradient problem and enable the training of deeper networks.

### Middle Section
The middle part is a bridge between the encoder and decoder sections. It further processes the feature maps from the encoder.

- `mid_conv1` to `mid_conv9`: These are convolutional blocks similar to the encoder, designed to further process the feature maps. The number of channels remains constant throughout these layers. This section can be simplified or expanded based on the complexity required.

### Decoder Section
The decoder part of the U-Net upsamples the feature maps and reduces the number of channels. It also concatenates the feature maps from the encoder using skip connections.

- `dec_conv8` to `dec_conv1`: Each `dec_conv` layer consists of two convolutional layers with batch normalization and ReLU activation. The number of channels is progressively reduced.
- `up7` to `up1`: These are upsampling layers that increase the spatial dimensions of the feature maps. They use bilinear interpolation for upsampling.
- The `torch.cat` operations in the decoder concatenate the upsampled features with the corresponding features from the encoder. This is a crucial part of U-Net, allowing the network to use both high-level and low-level features for reconstruction.

### Output Section
The final output is generated through a convolutional layer that maps the feature maps to the desired number of output channels (e.g., 3 for RGB images).

- `out_conv`: This layer uses a 1x1 convolution to reduce the number of output channels to match the number of channels in the target images. It is followed by a Sigmoid activation function to ensure the output values are in a valid range (e.g., [0, 1] for normalized images).

### Forward Function
This function defines the forward pass of the network. It sequentially applies all the layers and functions defined in the `__init__` method. The feature maps are processed through the encoder, middle, and decoder sections, and the final output is produced. The forward function ensures that the skip connections are correctly utilized by concatenating the encoder features with the corresponding decoder features.

This architecture is a standard U-Net model used for tasks like image segmentation and denoising. The use of residual connections and skip connections typically helps in training deeper models more effectively.


---

## ChatGPT Said...
Your code provides a comprehensive setup for training a U-Net model for image denoising. It includes data loading, model definition, training, validation, testing, and plotting of training/validation loss. The overall structure seems sound. Here are a few observations and potential improvements:

1. **Data Directories**: You've set the data directories (`train_dir`, `test_dir`, `val_dir`) as relative paths. Ensure that the `train`, `test`, and `val` directories with the subdirectories `noisy` and `clean` are present in the same directory where the script is executed.

2. **Model Checkpoint Loading**: Your script attempts to load a pre-trained model (`best_psnr_denocoder_pytorch.pth`). Ensure this file is in the directory where the script runs, or provide the correct path.

3. **Loss and Optimizer**: The combination of L1 and MSE loss is a standard choice for image reconstruction tasks. The optimizer settings (learning rate and weight decay) are also typical, but might require tuning based on your specific data.

4. **Early Stopping and Learning Rate Scheduler**: You've implemented early stopping and a learning rate scheduler, which are good practices for preventing overfitting and for adaptive learning rate adjustment.

5. **Data Augmentation**: You've commented out some data augmentation techniques (like `RandomColorJitterWithRandomFactors`). Consider if they are beneficial for your use case and uncomment them if needed.

6. **Plotting**: The script plots training and validation losses at the end, which is helpful for analyzing the training process.

7. **Saving Model Checkpoints**: The script saves the model with the best loss and PSNR. Ensure you have the necessary permissions to write to the disk.

8. **GPU Utilization**: The script checks for CUDA availability and moves the model to GPU if available. This is good practice for efficient training on compatible hardware.

9. **PSNR Calculation**: The PSNR function seems appropriately implemented for performance evaluation.

10. **Error Handling**: Consider adding error handling, especially for file and directory operations, and for data loading processes.

11. **Reproducibility**: Setting a manual seed (`torch.manual_seed(42)`) is good for reproducibility but remember that complete reproducibility in PyTorch also depends on other factors, especially when using GPUs.

12. **Batch Size and Epochs**: Ensure that the `batch_size` and `epochs` are set appropriately for your dataset size and hardware capabilities.

Remember to test your code with a subset of your data first to ensure everything is functioning as expected. Also, keep track of your system's resources (like memory and disk space) when running the training, especially with large datasets or for a large number of epochs.


##Example Script Outputs:
/test_these.py
PLEASE REFER to the test_these folder for inference.

/ztest_gpu.py
GPU: NVIDIA GeForce RTX 3060
tensor([5., 7., 9.], device='cuda:0')

/zcount_parameter.py
Number of trainable parameters: 1488915

/model.py"
No pre-trained model found. Training from scratch.
Epoch [1/48], Loss: 0.4590, PSNR: 9.0256
Validation Loss: 0.4111, PSNR: 9.7463
Epoch [2/48], Loss: 0.3906, PSNR: 10.1210
Validation Loss: 0.3656, PSNR: 10.5915
Epoch [3/48], Loss: 0.3631, PSNR: 10.6348
Validation Loss: 0.3496, PSNR: 10.9311
Epoch [4/48], Loss: 0.3445, PSNR: 11.0200
Validation Loss: 0.3428, PSNR: 11.0771
Epoch [5/48], Loss: 0.3273, PSNR: 11.3700
Validation Loss: 0.3229, PSNR: 11.5076
Epoch [6/48], Loss: 0.3136, PSNR: 11.6636
Validation Loss: 0.3014, PSNR: 11.9478
Epoch [7/48], Loss: 0.2995, PSNR: 11.9669
Validation Loss: 0.2823, PSNR: 12.3748
Epoch [8/48], Loss: 0.2856, PSNR: 12.2870
Validation Loss: 0.2868, PSNR: 12.3015
Epoch [9/48], Loss: 0.2760, PSNR: 12.5328
Validation Loss: 0.2594, PSNR: 12.9594
Epoch [10/48], Loss: 0.2644, PSNR: 12.8281
Validation Loss: 0.2585, PSNR: 13.0038
Epoch [11/48], Loss: 0.2539, PSNR: 13.1152
Validation Loss: 0.2548, PSNR: 13.1207
Epoch [12/48], Loss: 0.2438, PSNR: 13.3911
Validation Loss: 0.2331, PSNR: 13.7125
Epoch [13/48], Loss: 0.2349, PSNR: 13.6553
Validation Loss: 0.2268, PSNR: 13.9173
Epoch [14/48], Loss: 0.2251, PSNR: 13.9547
Validation Loss: 0.2127, PSNR: 14.3588
Epoch [15/48], Loss: 0.2171, PSNR: 14.2186
Validation Loss: 0.2056, PSNR: 14.6265
Epoch [16/48], Loss: 0.2083, PSNR: 14.5040
Validation Loss: 0.2001, PSNR: 14.7708
Epoch [17/48], Loss: 0.1996, PSNR: 14.8151
Validation Loss: 0.2000, PSNR: 14.8162
Epoch [18/48], Loss: 0.1921, PSNR: 15.0934
Validation Loss: 0.1840, PSNR: 15.3575
Epoch [19/48], Loss: 0.1849, PSNR: 15.3527
Validation Loss: 0.1888, PSNR: 15.1795
Epoch [20/48], Loss: 0.1779, PSNR: 15.6158
Validation Loss: 0.1744, PSNR: 15.8039
Epoch [21/48], Loss: 0.1722, PSNR: 15.8854
Validation Loss: 0.1675, PSNR: 16.1288
Epoch [22/48], Loss: 0.1649, PSNR: 16.1948
Validation Loss: 0.1599, PSNR: 16.4019
Epoch [23/48], Loss: 0.1585, PSNR: 16.5062
Validation Loss: 0.1549, PSNR: 16.7032
Epoch [24/48], Loss: 0.1544, PSNR: 16.7112
Validation Loss: 0.1499, PSNR: 17.0306
Epoch [25/48], Loss: 0.1469, PSNR: 17.0932
Validation Loss: 0.1374, PSNR: 17.5503
Epoch [26/48], Loss: 0.1417, PSNR: 17.3295
Validation Loss: 0.1383, PSNR: 17.5362
Epoch [27/48], Loss: 0.1368, PSNR: 17.6108
Validation Loss: 0.1281, PSNR: 18.0941
Epoch [28/48], Loss: 0.1323, PSNR: 17.8748
Validation Loss: 0.1219, PSNR: 18.4767
Epoch [29/48], Loss: 0.1278, PSNR: 18.1278
Validation Loss: 0.1273, PSNR: 18.2552
Epoch [30/48], Loss: 0.1240, PSNR: 18.3380
Validation Loss: 0.1200, PSNR: 18.6236
Epoch [31/48], Loss: 0.1210, PSNR: 18.5212
Validation Loss: 0.1208, PSNR: 18.5556
Epoch [32/48], Loss: 0.1163, PSNR: 18.8424
Validation Loss: 0.1123, PSNR: 19.1592
Epoch [33/48], Loss: 0.1123, PSNR: 19.0597
Validation Loss: 0.1119, PSNR: 19.1459
Epoch [34/48], Loss: 0.1074, PSNR: 19.3705
Validation Loss: 0.0977, PSNR: 20.0343
Epoch [35/48], Loss: 0.1042, PSNR: 19.5855
Validation Loss: 0.1026, PSNR: 19.7171
Epoch [36/48], Loss: 0.1008, PSNR: 19.8480
Validation Loss: 0.0978, PSNR: 20.1919
Epoch [37/48], Loss: 0.0995, PSNR: 19.8962
Validation Loss: 0.0949, PSNR: 20.2933
Epoch [38/48], Loss: 0.0963, PSNR: 20.1377
Validation Loss: 0.0900, PSNR: 20.6487
Epoch [39/48], Loss: 0.0927, PSNR: 20.4455
Validation Loss: 0.0891, PSNR: 20.8581
Epoch [40/48], Loss: 0.0908, PSNR: 20.5816
Validation Loss: 0.0875, PSNR: 20.9049
Epoch [41/48], Loss: 0.0889, PSNR: 20.7054
Validation Loss: 0.0820, PSNR: 21.2531
Epoch [42/48], Loss: 0.0861, PSNR: 20.9458
Validation Loss: 0.0850, PSNR: 21.2291
Epoch [43/48], Loss: 0.0846, PSNR: 21.0837
Validation Loss: 0.0804, PSNR: 21.5047
Epoch [44/48], Loss: 0.0830, PSNR: 21.1821
Validation Loss: 0.0805, PSNR: 21.6124
Epoch [45/48], Loss: 0.0802, PSNR: 21.4567
Validation Loss: 0.0785, PSNR: 21.7757
Epoch [46/48], Loss: 0.0794, PSNR: 21.5219
Validation Loss: 0.0734, PSNR: 22.1452
Epoch [47/48], Loss: 0.0793, PSNR: 21.5721
Validation Loss: 0.0736, PSNR: 22.1541
Epoch [48/48], Loss: 0.0774, PSNR: 21.6938
Validation Loss: 0.0686, PSNR: 22.5416
Test Loss: 0.0716, PSNR: 22.2195



# Image Processing Scripts
This directory contains a collection of Python scripts designed to assist in the management and preparation of image datasets for a larger project. The primary function of these scripts is to consolidate 'before' and 'after' images into the `resized_imgs` folder for resizing, editing, and additions. After you can redistribute them into `test`, `train`, and `validate` folders for further use.

## Overview of Scripts
1. **`zmoveimgs.py`**: This is the primary script used for MOVE OUT images into training, validation, and test datasets. It segregates the images from the `resized_imgs` folder into `train`, `val`, and `test` folders, each having `clean` and `noisy` subfolders.

2. **`zgetimgs.py`**: This script is designed for moving images BACK INTO the resized_imgs folder from the train, val, and test directories. It effectively reverses the actions of zmoveimgs.py. Each image pair, consisting of a 'before' image from the noisy subfolders and an 'after' image from the clean subfolders, is relocated back to the resized_imgs folder. This is particularly useful when you need to consolidate your dataset back into a single directory, either for additional preprocessing steps or for archiving.

3. **`zresizeimgs.py`**: Use this script to resize images in the `resized_imgs` folder. It should be run before `zmove.py` if you have added new jpeg images to the resized images directory to ensure all images are of uniform size.

4. **`zname.py` and `zrename.py`**: If you add new images to the data set you can consider using these. The name scripts are used for renaming images to match the standard convention fo the project. They should be used cautiously to avoid disrupting the naming convention of your dataset. It's recommended to make backups before using these scripts.


## Instructions
### Setup
Ensure that all images are initially placed in the `resized_imgs` directory. The directory structure for the `train`, `val`, and `test` folders will be created automatically by the scripts.

### Using `zresizeimgs.py`
Run this script to resize all images in the `resized_imgs` folder. This is an essential step to standardize image sizes across your dataset, and will throw an error if you during training if you have an image in which the aspect ratio is mismatched with the array being used for matrix multiplication. 

Simple Example: If we have a 2:3 and a 3:4 image, in terms of aspect ratio the dot products would mismatch as follows.....
    """
      2x3:               3x4:
      x-x-Error          x-x-x
      x-x-Error          x-x-x
      x-x-Error          x-x-x
      Error-Error-Error  x-x-x
    """
Note: Though this a simple example, a nugget to take away from this is that if we have the same aspect ratio across two different images we can normalize before the training step of 'model.py' or maintain consideration as to the intrinsic meaning of what we could accomplish with different size images and relative aspect ratios. (I have not been creative or logical enough to come up with a significant use case, but I am sure a very powerful one exists.)

### Using `zmove.py`
This script is the main tool for distributing images into appropriate datasets. It randomly assigns images to the `train`, `val`, and `test` folders based on specified proportions.

1. **Running the Script**: Simply execute `zmove.py`. The script will automatically shuffle and distribute the images.

2. **Folder Structure**:
   - `train/clean`
   - `train/noisy`
   - `val/clean`
   - `val/noisy`
   - `test/clean`
   - `test/noisy`

### Using `zname.py` and `zrename.py`
These scripts are used for renaming image files. They must be used with caution to maintain the integrity of your dataset.

1. **Backup First**: Always create a backup of your images before running these scripts.

2. **Testing on a Subset**: Try these scripts on a small set of images first to ensure they work as expected.

3. **Running the Scripts**: Use `zname.py` for initial naming and `zrename.py` for any subsequent renaming needs.

## Caution
- **Backup**: Always backup your images before using `zname.py` and `zrename.py`.
- **Testing**: Test the scripts on a small subset of your images first.
- **Order of Execution**: Run `zresizeimgs.py` before `zmove.py` to ensure all images are properly sized.

## Additional Notes
This folder is part of a larger project focused on image processing. The scripts are tailored for a specific workflow involving the preparation of image datasets. For any modifications or custom use-cases, please review and adjust the scripts accordingly.

---