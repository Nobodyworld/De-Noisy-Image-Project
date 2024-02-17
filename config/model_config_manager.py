import torch.nn as nn
import json

# Load the configuration file
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

# Dynamic layer creation mapping
LAYER_MAP = {
    "Conv2d": nn.Conv2d,
    "BatchNorm2d": nn.BatchNorm2d,
    "ReLU": nn.ReLU,
    "MaxPool2d": nn.MaxPool2d,
    "Upsample": nn.Upsample
}

# Function to create a single layer from the config
def create_layer(layer_config, in_channels):
    layer_type = layer_config['type']
    layer_class = LAYER_MAP[layer_type]
    
    if layer_type in ["Conv2d", "BatchNorm2d"]:
        out_channels = in_channels * layer_config.get('out_channels_factor', 1)
        return layer_class(in_channels, out_channels, **{k: v for k, v in layer_config.items() if k not in ["type", "out_channels_factor"]}), out_channels
    else:
        return layer_class(**{k: v for k, v in layer_config.items() if k != "type"}), in_channels
