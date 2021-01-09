# TODO rename to experiments.py
from os.path import join

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.pcutil import plot_3d_point_cloud


def fixed(full_model, device, dataset, results_dir, epoch,
          amount=30, mean=0.0, std=0.015, noises_per_item=50, triangulation_config={'execute': False, 'method': 'edge', 'depth': 2}):

    dataloader = DataLoader(dataset)

    for i, data in enumerate(dataloader):

        partial, _, _, idx = data
        partial = partial.to(device)

        from utils.util import show_3d_cloud

        for j in range(noises_per_item):
            fixed_noise = torch.zeros(1, 1024).normal_(mean=mean, std=std).to(device)

            reconstruction = full_model(partial, None, None, epoch, device, noise=fixed_noise)[0]
            reconstruction = reconstruction.cpu()

            np.save(join(results_dir, 'fixed', f'{i}_{j}_fixed_noise'), np.array(fixed_noise.cpu().numpy()))
            np.save(join(results_dir, 'fixed', f'{i}_{j}_reconstruction'), reconstruction.numpy())

            fig = plot_3d_point_cloud(reconstruction[0], reconstruction[1], reconstruction[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'fixed', f'{i}_{j}_fixed_reconstructed.png'))
            plt.close(fig)

            # dataset.inverse_scale_to_scene(idx, reconstruction.T.numpy())

            # np.save(join(results_dir, 'fixed', f'{i}_{j}_rescaled'), dataset.inverse_scale(idx, reconstruction.T.numpy())) # TODO extract to real data fixed experiment

        partial = partial.cpu()
        fig = plot_3d_point_cloud(partial[0][0], partial[0][1], partial[0][2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'fixed', f'{i}_partial.png'))
        plt.close(fig)

        np.save(join(results_dir, 'fixed', f'{i}_partial'), np.array(partial.cpu().numpy()))


experiment_functions_dict = {
    # 'interpolation': interpolation,
    # 'interpolation_between_two_points': interpolation_between_two_points,
    # 'reconstruction': reconstruction,
    # 'sphere': sphere,
    # 'sphere_triangles': sphere_triangles,
    # 'sphere_triangles_interpolation': sphere_triangles_interpolation,
    # 'different_number_of_points': different_number_of_points,
    'fixed': fixed,
}
