import os

def rename_files(directory, prefix='github_', suffix='_before', extension='.jpg'):
    count = 1
    # Sort files for consistent ordering
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(extension):
            new_filename = f"{prefix}{str(count).zfill(2)}{suffix}{extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            # Check if a file with the new name already exists
            if not os.path.exists(new_path):
                try:
                    os.rename(old_path, new_path)
                    print(f'Renamed {filename} to {new_filename}.')
                except Exception as e:
                    print(f'Error renaming {filename} to {new_filename}: {e}')
            else:
                print(f'File {new_filename} already exists. Skipping {filename}.')
            count += 1
        else:
            print(f'Ignored {filename} (not a {extension} file).')

# Set up paths based on the script's location
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
stage_data = os.path.join(project_root, "stage_data")

rename_files(stage_data)
