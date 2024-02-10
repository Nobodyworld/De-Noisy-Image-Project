# atest_count_param.py
import torch
import json
import os

source_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "model")

# Get the current script directory
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)

Unet = os.path.join(project_root, "path")

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

# Path to your config.json file
config_path = './config/config.json'  # Adjust the path according to your directory structure

# Load the configuration
config = load_config(config_path)

# Get the model name from config
UNet = config['model']['name'] 

# Initialize the model
model = UNet()

# Load the state dictionary from the saved model using the path from config.json
model_checkpoint_path = config['model']['path']  # Get the model checkpoint path from config
state_dict = torch.load(model_checkpoint_path)

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Now you can count the trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {total_params}")
