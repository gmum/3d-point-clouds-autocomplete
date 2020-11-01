from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, zeros_


class Softsign(nn.Module):
    def __init__(self, softening_value=1):
        super().__init__()
        self.softening_value = softening_value

    def forward(self, input):
        return input / (input.abs() + self.softening_value)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_epoch = 0

    def forward(self, *input):
        raise NotImplementedError

    # def optimize_parameters(self, *args):
    #     raise NotImplementedError
    #
    # def load_weights(self, weights_dir, epoch_n=None):
    #     raise NotImplementedError
    #
    # def save_weights(self, weights_dir, epoch_n=None):
    #     raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class PointNetClassification(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.n_classes = config['n_classes']
        self.pointnet = PointNet(config)
        self.classification = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=self.n_classes, bias=True)
        )

    def forward(self, pc):
        if self.pointnet.without_tnet:
            if self.pointnet.return_indices:
                global_features, point_indices = self.pointnet(pc)
                classification_output =  self.classification(global_features)
                return global_features, classification_output, point_indices
            else:
                global_features = self.pointnet(pc)
                classification_output =  self.classification(global_features)
                return global_features, classification_output
        else:
            if self.pointnet.return_indices:
                input_transform, features_transform, global_features, point_indices = self.pointnet(pc)
                classification_output =  self.classification(global_features)
                return input_transform, features_transform, global_features, classification_output, point_indices
            else:
                input_transform, features_transform, global_features = self.pointnet(pc)
                classification_output =  self.classification(global_features)
                return input_transform, features_transform, global_features, classification_output


class PointNet(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']  # Default PointNet z_size=1024
        self.binary_point_net = False # (('distribution' in config['z']) and (config['z']['distribution'].lower() in ('beta', 'bernoulli')))
        self.binary_activation = None # config['z']['binary_activation'].lower() if 'binary_activation' in config['z'] else None

        # self.train_tnet3 = 'train_tnet' not in config or config['train_net']
        self.without_tnet = 'no_tnet' in config and config['no_tnet']
        self.tnet_weights = config['tnet_weights'] if 'tnet_weights' in config else None
        self.self_supervised_weights = config['pointnet_selfsup_weights'] if 'pointnet_selfsup_weights' in config else None
        self.n_neighbours = config['n_neighbours'] if 'n_neighbours' in config else 1

        if not self.without_tnet:
            self.tnet3 = TNet(k=3)

        self.PN1 = nn.Sequential(
            nn.Conv1d(3 * self.n_neighbours, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        if not self.without_tnet:
            self.tnet64 = TNet(k=64)

        self.PN2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, self.z_size, 1),
            nn.BatchNorm1d(self.z_size),
        )

        self.last_activation = self._get_correct_last_activation()
        # This is usually not set to True, only for critical points visualization.
        self.return_indices = 'return_indices' in config and config['return_indices']

        self.aggregate_function = config['permutation_invariant_function'] if 'permutation_invariant_function' in config else 'max'
        if self.aggregate_function == 'sum':
            if self.return_indices:
                raise ValueError(f'Aggregate function `{self.aggregate_function}` does not return indices.')
            self.permutation_invariant_fn = torch.sum
        elif self.aggregate_function == 'mean':
            if self.return_indices:
                raise ValueError(f'Aggregate function `{self.aggregate_function}` does not return indices.')
            self.permutation_invariant_fn = torch.mean
        elif self.aggregate_function == 'max':
            if self.return_indices:
                def max_values(features: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
                    max_ = torch.max(features, dim=dim)
                    return max_.indices, max_.values
            else:
                def max_values(features: torch.Tensor, dim: int) -> torch.Tensor:
                    return torch.max(features, dim=dim).values

            self.permutation_invariant_fn = max_values
        elif self.aggregate_function == 'min':
            if self.return_indices:
                def min_values(features: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
                    min_ = torch.min(features, dim=dim)
                    return min_.indices, min_.values
            else:
                def min_values(features: torch.Tensor, dim: int) -> torch.Tensor:
                    return torch.min(features, dim=dim).values

            self.permutation_invariant_fn = min_values
        elif self.aggregate_function == 'median':
            if self.return_indices:
                def median_values(features: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
                    median = torch.median(features, dim=dim)
                    return median.indices, median.values
            else:
                def median_values(features: torch.Tensor, dim: int) -> torch.Tensor:
                    return torch.median(features, dim=dim).values

            self.permutation_invariant_fn = median_values
        elif self.aggregate_function.startswith('combine'):
            if self.return_indices:
                raise ValueError(f'Aggregate function `{self.aggregate_function}` does not return indices.')

            self.aggregation_network = nn.Sequential(
                nn.Linear(in_features=3 * self.z_size, out_features=2 * self.z_size, bias=True),
                nn.BatchNorm1d(2 * self.z_size),
                nn.ReLU(),

                nn.Linear(in_features=2 * self.z_size, out_features=self.z_size, bias=True),
                self._get_correct_last_activation()
            )

            if self.aggregate_function == 'combine':
                def get_all_aggregations(features: torch.Tensor, dim: int) -> torch.Tensor:
                    return self.aggregation_network(
                        torch.cat(
                            (torch.sum(features, dim=dim),
                             torch.mean(features, dim=dim),
                             torch.max(features, dim=dim).values),
                            dim=1
                        )
                    )
            elif self.aggregate_function == 'combine2':  # Great name
                def get_all_aggregations(features: torch.Tensor, dim: int) -> torch.Tensor:
                    return self.aggregation_network(
                        torch.cat(
                            (torch.median(features, dim=dim).values,
                             torch.mean(features, dim=dim),
                             torch.max(features, dim=dim).values),
                            dim=1
                        )
                    )
            else:
                raise ValueError(f'Wrong combination name. Got: `{self.aggregate_function}`')

            self.permutation_invariant_fn = get_all_aggregations
        elif self.aggregate_function.startswith('recurrent'):
            self.aggregation_network = nn.Sequential(
                nn.Linear(in_features=3 * self.z_size, out_features=2 * self.z_size, bias=True),
                nn.BatchNorm1d(2 * self.z_size),
                nn.ReLU(),

                nn.Linear(in_features=2 * self.z_size, out_features=self.z_size, bias=True),
                self._get_correct_last_activation()
            )

        elif self.aggregate_function.startswith('top'):
            k = int(self.aggregate_function[3:])

            self.aggregation_network = nn.Sequential(
                nn.Linear(in_features=k * self.z_size, out_features=self.z_size, bias=True),
                self._get_correct_last_activation()
            )

            if self.return_indices:
                def combine_topk_features(features: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
                    topk = features.topk(k=k, dim=dim)
                    return topk.indices.flatten(), self.aggregation_network(topk.values.view(len(features), -1))
            else:
                def combine_topk_features(features: torch.Tensor, dim: int) -> torch.Tensor:
                    return self.aggregation_network(features.topk(k=k, dim=dim).values.view(len(features), -1))

            self.permutation_invariant_fn = combine_topk_features
        else:
            raise ValueError('Invalid permutation invariant function')

        self.init_weights()

    def _get_correct_last_activation(self):
        if self.binary_point_net:
            last_activation = nn.Tanh() if self.binary_activation == 'tanh' else Softsign(softening_value=0.01)
        else:
            last_activation = nn.ReLU()

        return last_activation

    def init_weights(self):
        if self.self_supervised_weights:
            self.load_state_dict(torch.load(self.self_supervised_weights))
            return

        def weights_init(m):
            classname = m.__class__.__name__
            if classname in ('Conv1d', 'Linear'):
                kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    zeros_(m.bias)

        self.apply(weights_init)

        if self.binary_point_net:
            # Different initialization than in usual PointNet,
            # because last activation is Tanh
            kaiming_normal_(self.PN2[-2].weight, nonlinearity='tanh')

            if self.aggregate_function.startswith('combine'):
                kaiming_normal_(self.aggregation_network[-2].weight, nonlinearity='tanh')

        if self.without_tnet:
            return

        if self.tnet_weights:
            self.tnet3.load_state_dict(torch.load(self.tnet_weights))
        else:
            # Last layers of T-nets have special initialization procedure.
            # They are initialized to return identity matrices at the beggining.
            zeros_(self.tnet3.fc_part[-1].weight)
            self.tnet3.fc_part[-1].bias = nn.Parameter(torch.eye(3).flatten())

            zeros_(self.tnet64.fc_part[-1].weight)
            # normal_(self.tnet64.fc_part[-1].weight, std=0.01)
            self.tnet64.fc_part[-1].bias = nn.Parameter(torch.eye(64).flatten())

    def forward(self, pc):
        # Starting pc dimension is Nx3x2048
        if self.without_tnet:
            # Nx64x2048
            features = self.PN1(pc)
            # NxZ_SIZEx2048
            point_features = self.last_activation(self.PN2(features))
            # NxZ_SIZE
            if self.return_indices:
                point_indices, global_features = self.permutation_invariant_fn(point_features, dim=2)
                return global_features, point_indices
            else:
                global_features = self.permutation_invariant_fn(point_features, dim=2)
                return global_features
        else:
            # Nx3*(1+n_n)x2048
            input_transform = []
            for n in range(self.n_neighbours):
                input_transform.append(self.tnet3(pc[:, 3 * n: 3 * (n+1)]))
            input_transform = torch.cat(input_transform, dim=1)

            # Nx3*(1+n_n)x2048 = (Nx2048x3*(1+n_n) @ Nx3*(1+n_n)x3*(1+n_n))^T
            pc_transformed = torch.bmm(pc.transpose(1, 2), input_transform).transpose(1, 2)

            # Nx64x2048
            features = self.PN1(pc_transformed)
            # Nx64x64
            features_transform = self.tnet64(features)
            # Nx64x2048 = (Nx2048x64 @ Nx64x64)^T
            features_transformed = torch.bmm(features.transpose(1, 2), features_transform).transpose(1, 2)

            # NxZ_SIZEx2048
            point_features = self.last_activation(self.PN2(features_transformed))
            # NxZ_SIZE

            if self.return_indices:
                point_indices, global_features = self.permutation_invariant_fn(point_features, dim=2)
                return input_transform, features_transform, global_features, point_indices
            else:
                global_features = self.permutation_invariant_fn(point_features, dim=2)
                return input_transform, features_transform, global_features


class TNet(BaseModel):
    def __init__(self, k):
        super().__init__()

        self.k = k

        self.conv_part = nn.Sequential(
            nn.Conv1d(self.k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.MaxPool2d([1, 2048])
        )

        self.fc_part = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, self.k * self.k)
        )

    def forward(self, pc):
        step_1 = self.conv_part(pc).view(len(pc), -1)
        step_2 = self.fc_part(step_1)
        return step_2.view(-1, self.k, self.k)
