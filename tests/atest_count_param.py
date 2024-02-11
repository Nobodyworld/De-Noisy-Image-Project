import torch
import json
import os
from pathlib import Path
import sys

# Add the models directory to sys.path to make the unet module available
current_dir = Path(__file__).parent
root_dir = current_dir.parent
models_dir = root_dir / 'models'
sys.path.insert(0, str(models_dir))

from unet import UNet  # Now you can directly import UNet

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

# Assuming this script is at the root of your project structure where config.json is also located
config_path = os.path.join(root_dir, 'config/config.json')  # Adjust this if your config.json is located elsewhere

# Load the configuration
config = load_config(config_path)

# Initialize the model
model = UNet()

# Load the state dictionary from the saved model using the path from config.json
model_checkpoint_path = config['model']['path']  # Get the model checkpoint path from config

# Assuming the model path is relative to the config.json's location
model_checkpoint_full_path = os.path.join(root_dir, model_checkpoint_path)

state_dict = torch.load(model_checkpoint_full_path, map_location=torch.device('cpu'))

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Now you can count the trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {total_params}")
