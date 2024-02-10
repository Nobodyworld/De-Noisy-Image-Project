import os
from PIL import Image

def find_images_with_different_size(root_dir, width=640, height=960):
    # Correctly unpack the values yielded by os.walk()
    for root, _, files in os.walk(root_dir):  # The underscore '_' is used to ignore the 'dirnames' value
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        if img.size != (width, height):
                            print(file_path)
                except IOError:
                    print(f"Could not open {file_path}")

if __name__ == "__main__":
    directory = "./data"  # Make sure this path is correct for your system
    find_images_with_different_size(directory)

