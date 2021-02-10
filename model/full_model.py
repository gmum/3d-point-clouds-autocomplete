from itertools import chain
from typing import Iterator

import torch
import torch.nn as nn
from torch.nn import Parameter

from model.encoder import Encoder
from model.hyper_network import HyperNetwork
from model.target_network import TargetNetwork
from utils.points import generate_points


class FullModel(nn.Module):

    @staticmethod
    def _complete_config(config):
        config['hyper_network']['target_network_layer_out_channels'] = config['target_network']['layer_out_channels']
        config['hyper_network']['target_network_use_bias'] = config['target_network']['use_bias']
        config['hyper_network']['input_size'] = config['random_encoder']['output_size'] + \
                                                config['real_encoder']['output_size']

        config['hyper_network']['target_network_freeze_layers_learning'] = config['target_network'][
            'freeze_layers_learning']

    def get_noise_size(self):
        return self.random_encoder_output_size

    def _resolve_mode(self, config):
        self.random_encoder_output_size = config['random_encoder']['output_size']
        if config['random_encoder']['output_size'] > 0 and config['real_encoder']['output_size'] > 0:
            self.mode = HyperPocket()
            self.random_encoder = Encoder(config['random_encoder'], is_vae=True)
            self.real_encoder = Encoder(config['real_encoder'], is_vae=False)
        elif config['random_encoder']['output_size'] > 0:
            self.mode = HyperCloud()
            self.random_encoder = Encoder(config['random_encoder'], is_vae=True)
        elif config['real_encoder']['output_size'] > 0:
            self.mode = HyperRec()
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

    def forward(self, existing, missing, gt_shape, epoch, device, noise=None):

        if existing.size(-1) == 3:
            existing.transpose_(existing.dim() - 2, existing.dim() - 1)

        if noise is None and missing is not None and missing.size(-1) == 3:
            missing.transpose_(missing.dim() - 2, missing.dim() - 1)

        if gt_shape[-1] == 3:
            gt_shape[1], gt_shape[2] = gt_shape[2], gt_shape[1]

        latent, mu, logvar = self.mode.get_latent(self, existing, missing, noise)

        target_networks_weights = self.hyper_network(latent)
        reconstruction = torch.zeros(gt_shape).to(device)

        for j, target_network_weights in enumerate(target_networks_weights):
            target_network = TargetNetwork(self.target_network_config, target_network_weights).to(device)
            target_network_input = generate_points(config=self.point_generator_config, epoch=epoch,
                                                   size=(gt_shape[2], gt_shape[1]))
            reconstruction[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

        # reconstruction shape [BATCH_SIZE, 3, N]
        if self.training:
            return reconstruction, logvar, mu
        else:
            return reconstruction  # , latent, target_networks_weights

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.mode.get_parameters(self)


class ModelMode(object):

    def get_latent(self, model: FullModel, existing, missing, noise=None):
        raise NotImplementedError

    def get_parameters(self, model: FullModel) -> Iterator[Parameter]:
        raise NotImplementedError

    def has_generativity(self) -> bool:
        raise NotImplementedError


class HyperPocket(ModelMode):

    def get_latent(self, model: FullModel, existing, missing, noise=None):
        if model.training:
            codes, mu, logvar = model.random_encoder(missing)
            real_mu = model.real_encoder(existing)
            latent = torch.cat([codes, real_mu], 1)
            return latent, mu, logvar
        else:
            if noise is None:
                _, random_mu, _ = model.random_encoder(missing)
            else:
                random_mu = noise
            real_mu = model.real_encoder(existing)
            latent = torch.cat([random_mu, real_mu], 1)
            return latent, None, None

    def get_parameters(self, model: FullModel):
        return chain(model.random_encoder.parameters(),
                     model.real_encoder.parameters(),
                     model.hyper_network.parameters())

    def has_generativity(self) -> bool:
        return True


class HyperRec(ModelMode):

    def get_latent(self, model: FullModel, existing, missing, noise=None):
        return model.real_encoder(existing), None, None

    def get_parameters(self, model: FullModel):
        return chain(model.real_encoder.parameters(), model.hyper_network.parameters())

    def has_generativity(self) -> bool:
        return False


class HyperCloud(ModelMode):

    def get_latent(self, model: FullModel, existing, missing, noise=None):
        if model.training:
            return model.random_encoder(existing)
        else:
            if noise is None:
                _, random_mu, _ = model.random_encoder(existing)
            else:
                random_mu = noise
            return random_mu, None, None

    def get_parameters(self, model: FullModel):
        return chain(model.random_encoder.parameters(), model.hyper_network.parameters())

    def has_generativity(self) -> bool:
        return False
