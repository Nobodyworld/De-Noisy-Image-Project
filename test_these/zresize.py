import os
from PIL import Image

# Set image dimensions
img_height = 1920
img_width = 1280

# Define data directory
input_dir = '.\test_these'

def orient_to_portrait(image):
    if image.width > image.height:
        image = image.rotate(90, expand=True)
    return image

def resize_image(image, target_size):
    resized_image = image.resize(target_size, Image.LANCZOS)
    return resized_image

# Resize images in the same directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
        # Open the image and check if it has a different size
        image_path = os.path.join(input_dir, filename)
        with Image.open(image_path) as image:
            # Orient the image to portrait if it's in landscape
            image = orient_to_portrait(image)
            width, height = image.size
            if width != img_width or height != img_height:
                # Resize the image and save it to the same directory with the same filename
                image_resized = resize_image(image, (img_width, img_height))
                image_resized.save(image_path, quality=100)
                print(f'Resized {filename}.')
            else:
                print(f'Skipped {filename} (same dimensions).')
        del image  # Release the image handle
    else:
        print(f'Ignored {filename} (not an image file).')
