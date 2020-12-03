import torch
import torch.nn as nn


class HyperNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_size = config['input_size']
        self.use_bias = config['use_bias']
        self.relu_slope = config['relu_slope']
        # target network layers out channels
        target_network_out_ch = [3] + config['target_network_layer_out_channels'] + [3]
        target_network_use_bias = int(config['target_network_use_bias'])

        self.model = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048, bias=self.use_bias),
        )

        self.output = [
            nn.Linear(2048, (target_network_out_ch[x - 1] + target_network_use_bias) * target_network_out_ch[x],
                      bias=True)
            for x in range(1, len(target_network_out_ch))
        ]

        if not config['freeze_layers_learning']:
            self.output = nn.ModuleList(self.output)

    def forward(self, x):
        output = self.model(x)
        return torch.cat([target_network_layer(output) for target_network_layer in self.output], 1)
