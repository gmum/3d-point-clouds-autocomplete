import torch
import numpy as np


def generate_points_from_uniform_distribution(size, low=-1, high=1):
    while True:
        points = torch.zeros([size[0] * 3, *size[1:]]).uniform_(low, high)
        points = points[torch.norm(points, dim=1) < 1]
        if points.shape[0] >= size[0]:
            return points[:size[0]]


def generate_points(config, epoch, size, normalize_points=None):
    if normalize_points is None:
        normalize_points = config['target_network_input']['normalization']['enable']

    if normalize_points and config['target_network_input']['normalization']['type'] == 'progressive':
        normalization_max_epoch = config['target_network_input']['normalization']['epoch']

        normalization_coef = np.linspace(0, 1, normalization_max_epoch)[epoch - 1] \
            if epoch <= normalization_max_epoch else 1
        points = generate_points_from_uniform_distribution(size=size)
        points[np.linalg.norm(points, axis=1) < normalization_coef] = \
            normalization_coef * (
                    points[
                        np.linalg.norm(points, axis=1) < normalization_coef].T /
                    torch.from_numpy(
                        np.linalg.norm(points[np.linalg.norm(points, axis=1) < normalization_coef], axis=1)).float()
            ).T
    else:
        points = generate_points_from_uniform_distribution(size=size)

    return points
