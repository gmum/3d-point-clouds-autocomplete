import torch
import torch.nn as nn

from model.encoder import Encoder
from model.hyper_network import HyperNetwork
from model.target_network import TargetNetwork
from utils.points import generate_points


class FullModel(nn.Module):

    def _complete_config(self, config):
        config['hyper_network']['target_network_layer_out_channels'] = config['target_network']['layer_out_channels']
        config['hyper_network']['target_network_use_bias'] = config['target_network']['use_bias']
        config['hyper_network']['input_size'] = config['random_encoder']['output_size'] + \
                                                config['real_encoder']['output_size']

        config['hyper_network']['target_network_freeze_layers_learning'] = config['target_network']['freeze_layers_learning']

    def __init__(self, config):
        super().__init__()
        self._complete_config(config)

        self.random_encoder = Encoder(config['random_encoder'], is_vae=True)
        self.real_encoder = Encoder(config['real_encoder'], is_vae=False)
        self.hyper_network = HyperNetwork(config['hyper_network'])
        self.target_network_config = config['target_network']

        self.point_generator_config = {'target_network_input': config['target_network_input']}

    def forward(self, partial, gt, epoch, device):

        if partial.size(-1) == 3:
            partial.transpose_(partial.dim() - 2, partial.dim() - 1)

        if gt.size(-1) == 3:
            gt.transpose_(gt.dim() - 2, gt.dim() - 1)

        if self.training:
            codes, mu, logvar = self.random_encoder(partial)
            real_mu = self.real_encoder(partial)

            target_networks_weights = self.hyper_network(torch.cat([codes, real_mu], 1))
        else:
            _, random_mu, _ = self.random_encoder(partial)
            real_mu = self.real_encoder(partial)
            target_networks_weights = self.hyper_network(torch.cat([random_mu, real_mu], 1))

        reconstruction = torch.zeros(gt.shape).to(device)
        for j, target_network_weights in enumerate(target_networks_weights):
            target_network = TargetNetwork(self.target_network_config, target_network_weights).to(device)

            target_network_input = generate_points(config=self.point_generator_config, epoch=epoch,
                                                   size=(gt.shape[2], gt.shape[1]))
            reconstruction[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

        if self.training:
            return reconstruction, logvar, mu
        else:
            return reconstruction
