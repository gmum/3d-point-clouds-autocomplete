import torch
import torch.nn as nn


class HyperNetwork(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.input_size = config['remaining_size'] + config['real_size']
        self.use_bias = config['model']['HN']['use_bias']
        self.relu_slope = config['model']['HN']['relu_slope']
        # target network layers out channels
        target_network_out_ch = [3] + config['model']['TN']['layer_out_channels'] + [3]
        target_network_use_bias = int(config['model']['TN']['use_bias'])

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
                      bias=True).to(device)
            for x in range(1, len(target_network_out_ch))
        ]

        if not config['model']['TN']['freeze_layers_learning']:
            self.output = nn.ModuleList(self.output)

    def forward(self, x):
        output = self.model(x)
        return torch.cat([target_network_layer(output) for target_network_layer in self.output], 1)


class TargetNetwork(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.z_size = config['remaining_size']
        self.use_bias = config['model']['TN']['use_bias']
        # target network layers out channels
        out_ch = config['model']['TN']['layer_out_channels']

        layer_data, split_index = self._get_layer_data(start_index=0, end_index=out_ch[0] * 3,
                                                       shape=(out_ch[0], 3), weights=weights)
        self.layers = {"1": layer_data}

        for x in range(1, len(out_ch)):
            layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                           end_index=split_index + (out_ch[x - 1] * out_ch[x]),
                                                           shape=(out_ch[x], out_ch[x - 1]), weights=weights)
            self.layers[str(x + 1)] = layer_data

        layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                       end_index=split_index + (out_ch[-1] * 3),
                                                       shape=(3, out_ch[-1]), weights=weights)
        self.output = layer_data
        self.activation = torch.nn.ReLU()
        assert split_index == len(weights)

    def forward(self, x):
        for layer_index in self.layers:
            x = torch.mm(x, torch.transpose(self.layers[layer_index]["weight"], 0, 1))
            if self.use_bias:
                assert "bias" in self.layers[layer_index]
                x = x + self.layers[layer_index]["bias"]
            x = self.activation(x)
        return torch.mm(x, torch.transpose(self.output["weight"], 0, 1)) + self.output.get("bias", 0)

    def _get_layer_data(self, start_index, end_index, shape, weights):
        layer_data = {"weight": weights[start_index:end_index].view(shape[0], shape[1])}
        if self.use_bias:
            layer_data["bias"] = weights[end_index:end_index + shape[0]]
            end_index = end_index + shape[0]
        return layer_data, end_index


class EncoderForRandomPoints(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['remaining_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(512, self.z_size, bias=True)
        self.std_layer = nn.Linear(512, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class EncoderForRealPoints(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.z_size = config['real_size']
        self.use_bias = config['model']['ER']['use_bias']
        self.relu_slope = config['model']['ER']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=self.use_bias),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True)
        )
        self.mu_layer = nn.Linear(512, self.z_size, bias=True)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        return mu


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['remaining_size']
        self.use_bias = config['model']['D']['use_bias']
        self.relu_slope = config['model']['D']['relu_slope']
        self.dropout = config['model']['D']['dropout']

        self.model = nn.Sequential(
            nn.Linear(self.z_size, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 128, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1, bias=True)
        )

    def forward(self, x):
        logit = self.model(x)
        return logit
