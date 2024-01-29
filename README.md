# De-Noisy-Image-Project
This repository contains scripts and directories for a public domain image editing neural network, 'De-Noisy Image Project', using a U-Net architecture.

The main components are the `resized_imgs` folder for storing and preprocessing images before training, `test_these` for extra inference, and the `model.py` script for training and architecture. The `model.py` will run without an NVIDIA GPU but you should consider reducing the image sizes, the number of layers, or these varaibles below from near lines 190-200 of `model.py`.

  """
    # Set batch size, image dimensions
    batch_size = 2
    img_height = 1920
    img_width = 1280
    epochs = 32
  """

Note: As you can see above I am pushing my limits with the current architecture by using such large images. Using a batch size of two does ultimately get me where I need to be when holding that many numbers in my GPU at a single time. (1920x1280 x {i} = 2,457,600) Upon using the script, `/zcount_parameter.py` I can find that the the total avaliable parameters i have for this trained model is, "Number of trainable parameters: 5,158,995." Which as a whole number has to be 2. 

Batching Note: Having a batch size of two does not mean that my model only sees two images per 'epoch' but rather that my model is fed two images at a time, of all images in the dataset, until it finishes, which completes one epoch. So if I have 20 images I will then have 10 batches per epoch.

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
Number of trainable parameters: 5158995

/model.py"
No pre-trained model found. Training from scratch.
Epoch [1/32], Loss: 0.4423, PSNR: 9.2922
Validation Loss: 0.4056, PSNR: 9.9336
Epoch [2/32], Loss: 0.3627, PSNR: 10.6420
Validation Loss: 0.3549, PSNR: 10.8065
Epoch [3/32], Loss: 0.3114, PSNR: 11.7114
Validation Loss: 0.3065, PSNR: 11.7591
Epoch [4/32], Loss: 0.2657, PSNR: 12.8736
Validation Loss: 0.2396, PSNR: 13.6222
Epoch [5/32], Loss: 0.2262, PSNR: 14.0401
Validation Loss: 0.2355, PSNR: 13.8352
Epoch [6/32], Loss: 0.1901, PSNR: 15.3750
Validation Loss: 0.2652, PSNR: 13.7730
Epoch [7/32], Loss: 0.1618, PSNR: 16.6132
Validation Loss: 0.1581, PSNR: 16.7714
Epoch [8/32], Loss: 0.1367, PSNR: 17.8996
Validation Loss: 0.1240, PSNR: 18.7317
Epoch [9/32], Loss: 0.1165, PSNR: 19.1519
Validation Loss: 0.1109, PSNR: 19.6430
Epoch [10/32], Loss: 0.1017, PSNR: 20.1655
Validation Loss: 0.1003, PSNR: 20.4135
Epoch [11/32], Loss: 0.0908, PSNR: 21.0079
Validation Loss: 0.0894, PSNR: 20.7365
Epoch [12/32], Loss: 0.0822, PSNR: 21.7378
Validation Loss: 0.0776, PSNR: 22.5518
Epoch [13/32], Loss: 0.0758, PSNR: 22.3025
Validation Loss: 0.0747, PSNR: 22.7752
Epoch [14/32], Loss: 0.0709, PSNR: 22.7352
Validation Loss: 0.0700, PSNR: 23.1187
Epoch [15/32], Loss: 0.0685, PSNR: 23.0095
Validation Loss: 0.0610, PSNR: 23.7705
Epoch [16/32], Loss: 0.0647, PSNR: 23.3626
Validation Loss: 0.0593, PSNR: 24.3658
Epoch [17/32], Loss: 0.0622, PSNR: 23.6591
Validation Loss: 0.0586, PSNR: 24.3017
Epoch [18/32], Loss: 0.0622, PSNR: 23.6446
Validation Loss: 0.0538, PSNR: 24.8763
Epoch [19/32], Loss: 0.0596, PSNR: 23.9008
Validation Loss: 0.0531, PSNR: 24.9018
Epoch [20/32], Loss: 0.0599, PSNR: 23.9102
Validation Loss: 0.0560, PSNR: 24.9599
Epoch [21/32], Loss: 0.0579, PSNR: 24.1470
Validation Loss: 0.0652, PSNR: 23.6269
Epoch [22/32], Loss: 0.0573, PSNR: 24.1552
Validation Loss: 0.0511, PSNR: 25.0166
Epoch [23/32], Loss: 0.0573, PSNR: 24.2055
Validation Loss: 0.0507, PSNR: 25.2434
Epoch [24/32], Loss: 0.0561, PSNR: 24.3089
Validation Loss: 0.0524, PSNR: 25.0755
Epoch [25/32], Loss: 0.0584, PSNR: 24.1095
Validation Loss: 0.0537, PSNR: 24.7750
Epoch [26/32], Loss: 0.0559, PSNR: 24.3107
Validation Loss: 0.0493, PSNR: 25.4217
Epoch [27/32], Loss: 0.0567, PSNR: 24.2692
Validation Loss: 0.0526, PSNR: 25.4229
Epoch [28/32], Loss: 0.0536, PSNR: 24.6227
Validation Loss: 0.0528, PSNR: 25.2008
Epoch [29/32], Loss: 0.0540, PSNR: 24.5950
Validation Loss: 0.0492, PSNR: 25.5453
Epoch [30/32], Loss: 0.0533, PSNR: 24.6575
Validation Loss: 0.0568, PSNR: 24.3292
Epoch [31/32], Loss: 0.0551, PSNR: 24.5233
Validation Loss: 0.0532, PSNR: 25.0379
Epoch [32/32], Loss: 0.0541, PSNR: 24.6063
Validation Loss: 0.0541, PSNR: 25.2656
Test Loss: 0.0580, PSNR: 24.9193
