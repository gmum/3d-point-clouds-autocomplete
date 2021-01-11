# TODO rename to experiments.py
import json
from os.path import join

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.utils.dataset_generator import HyperPlane
from losses.champfer_loss import ChamferLoss
from utils.metrics import compute_all_metrics, jsd_between_point_cloud_sets
from utils.pcutil import plot_3d_point_cloud
from utils.util import show_3d_cloud, resample_pcd


def fixed(full_model, device, dataset, results_dir, epoch, amount=30, mean=0.0, std=0.015, noises_per_item=50,
          triangulation_config={'execute': False, 'method': 'edge', 'depth': 2}):
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

            fig = plot_3d_point_cloud(reconstruction[0], reconstruction[1], reconstruction[2], in_u_sphere=True,
                                      show=False)
            fig.savefig(join(results_dir, 'fixed', f'{i}_{j}_fixed_reconstructed.png'))
            plt.close(fig)

            # dataset.inverse_scale_to_scene(idx, reconstruction.T.numpy())

            # np.save(join(results_dir, 'fixed', f'{i}_{j}_rescaled'), dataset.inverse_scale(idx, reconstruction.T.numpy())) # TODO extract to real data fixed experiment

        partial = partial.cpu()
        fig = plot_3d_point_cloud(partial[0][0], partial[0][1], partial[0][2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'fixed', f'{i}_partial.png'))
        plt.close(fig)

        np.save(join(results_dir, 'fixed', f'{i}_partial'), np.array(partial.cpu().numpy()))


def evaluate_generativity(full_model, device, datasets_dict, results_dir, epoch, batch_size, num_workers,
                          mean=0.0, std=0.015):
    dataloaders_dict = {cat_name: DataLoader(cat_ds, pin_memory=True, batch_size=1, num_workers=num_workers)
                        for cat_name, cat_ds in datasets_dict.items()}
    chamfer_loss = ChamferLoss().to(device)

    with torch.no_grad():
        results = {}

        plane_points = np.zeros((3, 3))
        plane_points[1][2] = 1
        plane_points[2][0] = 1

        for cat_name, dl in dataloaders_dict.items():
            cat_gt = []
            for data in dl:
                _, remaining, _, _ = data
                remaining = remaining.to(device)
                cat_gt.append(remaining)
            cat_gt = torch.cat(cat_gt).contiguous()

            cat_results = {}

            for data in tqdm(dl, total=len(dl)):
                partial, _, _, _ = data
                partial = partial.to(device)

                obj_recs = []

                for j in range(len(cat_gt)):
                    fixed_noise = torch.zeros(1, 1024).normal_(mean=mean, std=std).to(device)
                    reconstruction = full_model(partial, None, None, epoch, device, noise=fixed_noise)

                    pc = reconstruction.cpu().detach().numpy()[0]
                    obj_recs.append(torch.from_numpy(pc.T[pc[1].argsort()[:1024]]).unsqueeze(0).to(device))

                obj_recs = torch.cat(obj_recs)

                for k, v in compute_all_metrics(obj_recs, cat_gt, batch_size, chamfer_loss).items():
                    cat_results[k] = cat_results.get(k, 0.0) + v.item()
                cat_results['jsd'] = cat_results.get('jsd', 0.0) + jsd_between_point_cloud_sets(
                    obj_recs.cpu().detach().numpy(), cat_gt.cpu().numpy())
            results[cat_name] = cat_results
            print(cat_name, cat_results)

        with open(join(results_dir, 'evaluate_generativity', str(epoch) + 'eval_gen_by_cat.json'), mode='w') as f:
            json.dump(results, f)


experiment_functions_dict = {
    # 'interpolation': interpolation,
    # 'interpolation_between_two_points': interpolation_between_two_points,
    # 'reconstruction': reconstruction,
    # 'sphere': sphere,
    # 'sphere_triangles': sphere_triangles,
    # 'sphere_triangles_interpolation': sphere_triangles_interpolation,
    # 'different_number_of_points': different_number_of_points,
    'fixed': fixed,
    'evaluate_generativity': evaluate_generativity
}
