# Dynamic Encoder Module
import torch.nn as nn

class DynamicEncoder(nn.Module):
    def __init__(self, config_path):
        super(DynamicEncoder, self).__init__()
        self.layers = nn.ModuleList()
        config = load_config(config_path)
        
        in_channels = config['input_channels']
        for block in config['blocks']:
            if block['type'] == 'encoder':
                for layer_config in block['layers']:
                    layer, in_channels = create_layer(layer_config, in_channels)
                    self.layers.append(layer)
            # Handle other block types similarly

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs