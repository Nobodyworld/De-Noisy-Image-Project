import os

# Define data directory 
input_dir = 'test_these'

# Initialize counter variable
count = 1

# Rename each .jpg file to have the format 'test_01_before', 'test_02_before', etc.
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):
        # Construct new filename
        new_filename = f'github_{str(count).zfill(2)}_before.jpg'
        # Rename the file
        os.rename(os.path.join(input_dir, filename), os.path.join(input_dir, new_filename))
        print(f'Renamed {filename} to {new_filename}.')
        count += 1
    else:
        print(f'Ignored {filename} (not a .jpg file).')
