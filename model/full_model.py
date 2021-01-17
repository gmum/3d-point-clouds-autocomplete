from itertools import chain
from typing import Iterator
from enum import Enum

import torch
import torch.nn as nn
from torch.nn import Parameter

from model.encoder import Encoder
from model.hyper_network import HyperNetwork
from model.target_network import TargetNetwork
from utils.points import generate_points


class FullModel(nn.Module):
    class Mode(Enum):
        DOUBLE_ENCODER = 1
        RANDOM_ENCODER = 2
        REAL_ENCODER = 3

    def _complete_config(self, config):
        config['hyper_network']['target_network_layer_out_channels'] = config['target_network']['layer_out_channels']
        config['hyper_network']['target_network_use_bias'] = config['target_network']['use_bias']
        config['hyper_network']['input_size'] = config['random_encoder']['output_size'] + \
                                                config['real_encoder']['output_size']

        config['hyper_network']['target_network_freeze_layers_learning'] = config['target_network'][
            'freeze_layers_learning']

    def _resolve_mode(self, config):
        if config['random_encoder']['output_size'] > 0 and config['real_encoder']['output_size'] > 0:
            self.model_mode = FullModel.Mode.DOUBLE_ENCODER
            self.random_encoder = Encoder(config['random_encoder'], is_vae=True)
            self.real_encoder = Encoder(config['real_encoder'], is_vae=False)
        elif config['random_encoder']['output_size'] > 0:
            self.model_mode = FullModel.Mode.RANDOM_ENCODER
            self.random_encoder = Encoder(config['random_encoder'], is_vae=True)
        elif config['real_encoder']['output_size'] > 0:
            self.model_mode = FullModel.Mode.REAL_ENCODER
            self.real_encoder = Encoder(config['real_encoder'], is_vae=False)
        else:
            raise ValueError("at least one encoder should have non zero output")

    def __init__(self, config):
        super().__init__()
        self._complete_config(config)
        self._resolve_mode(config)

        self.hyper_network = HyperNetwork(config['hyper_network'])
        self.target_network_config = config['target_network']

        self.point_generator_config = {'target_network_input': config['target_network_input']}

    def _get_latent(self, partial, remaining, noise=None):
        # TODO think about refactor (extract to he new class ModelStrategy)
        if self.model_mode == FullModel.Mode.DOUBLE_ENCODER:
            if self.training:
                codes, mu, logvar = self.random_encoder(remaining)
                real_mu = self.real_encoder(partial)
                latent = torch.cat([codes, real_mu], 1)
                return latent, mu, logvar
            else:
                if noise is None:
                    _, random_mu, _ = self.random_encoder(remaining)
                else:
                    random_mu = noise
                real_mu = self.real_encoder(partial)
                latent = torch.cat([random_mu, real_mu], 1)
                return latent, None, None
        elif self.model_mode == FullModel.Mode.RANDOM_ENCODER:
            if self.training:
                return self.random_encoder(partial)
            else:
                if noise is None:
                    _, random_mu, _ = self.random_encoder(partial)
                else:
                    random_mu = noise
                return random_mu, None, None
        elif self.model_mode == FullModel.Mode.REAL_ENCODER:
            return self.real_encoder(partial), None, None

    def forward(self, partial, remaining, gt, epoch, device, noise=None):

        if partial.size(-1) == 3:
            partial.transpose_(partial.dim() - 2, partial.dim() - 1)

        if noise is None and remaining.size(-1) == 3:
            remaining.transpose_(remaining.dim() - 2, remaining.dim() - 1)

        if noise is None and gt.size(-1) == 3:
            gt.transpose_(gt.dim() - 2, gt.dim() - 1)
        else:
            gt = torch.zeros([1, 3, 2048])  # TODO change gt to gt_shape

        latent, mu, logvar = self._get_latent(partial, remaining, noise)

        target_networks_weights = self.hyper_network(latent)
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

    params_lambdas = {
        Mode.DOUBLE_ENCODER: lambda self: chain(self.random_encoder.parameters(),
                                                self.real_encoder.parameters(),
                                                self.hyper_network.parameters()),
        Mode.RANDOM_ENCODER: lambda self: chain(self.random_encoder.parameters(),
                                                self.hyper_network.parameters()),
        Mode.REAL_ENCODER: lambda self: chain(self.real_encoder.parameters(),
                                              self.hyper_network.parameters()),
    }

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.params_lambdas[self.model_mode](self)
