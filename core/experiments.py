# TODO rename to experiments.py
import glob
import os
import shutil
from datetime import datetime
import json
from os.path import join, basename
from zipfile import ZipFile

import h5py
from sklearn import manifold
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.utils.dataset_generator import SlicedDatasetGenerator
from losses.champfer_loss import ChamferLoss
from model.full_model import FullModel
from utils.metrics import compute_all_metrics, jsd_between_point_cloud_sets
from utils.pcutil import plot_3d_point_cloud


def fixed(full_model: FullModel, device, datasets_dict, results_dir: str, epoch, amount=30, mean=0.0, std=0.015,
          noises_per_item=10, batch_size=8, save_plots=False,
          triangulation_config={'execute': False, 'method': 'edge', 'depth': 2}):
    # clean dir
    shutil.rmtree(join(results_dir, 'fixed'), ignore_errors=True)
    os.makedirs(join(results_dir, 'fixed'), exist_ok=True)

    dataloaders_dict = {cat_name: DataLoader(cat_ds, pin_memory=True, batch_size=batch_size)
                        for cat_name, cat_ds in datasets_dict.items()}
    for cat_name, dl in dataloaders_dict.items():

        for i, data in tqdm(enumerate(dl), total=len(dl)):

            existing, _, _, idx = data
            existing = existing.to(device)

            for j in range(noises_per_item):
                fixed_noise = torch.zeros(existing.shape[0], full_model.get_noise_size()).normal_(mean=mean, std=std).to(
                    device)
                reconstruction = full_model(existing, None, [existing.shape[0], 2048, 3], epoch, device,
                                            noise=fixed_noise).cpu()
                for k in range(reconstruction.shape[0]):
                    np.save(join(results_dir, 'fixed', f'{cat_name}_{i * batch_size + k}_{j}_reconstruction'),
                            reconstruction[k].numpy())
                    if save_plots:
                        fig = plot_3d_point_cloud(reconstruction[k][0], reconstruction[k][1], reconstruction[k][2],
                                                  in_u_sphere=True, show=False)
                        fig.savefig(join(results_dir, 'fixed', f'{cat_name}_{i * batch_size + k}_{j}_fixed_reconstructed.png'))
                        plt.close(fig)
                    # np.save(join(results_dir, 'fixed', f'{i*batch_size+k}_{j}_fixed_noise'), np.array(fixed_noise[k].cpu().numpy()))

                # dataset.inverse_scale_to_scene(idx, reconstruction.T.numpy())

                # np.save(join(results_dir, 'fixed', f'{i}_{j}_rescaled'),
                #        dataset.inverse_scale(idx, reconstruction.T.numpy()))  # TODO extract to existing data fixed experiment

            existing = existing.cpu()
            for k in range(existing.shape[0]):
                np.save(join(results_dir, 'fixed', f'{cat_name}_{i * batch_size + k}_existing'), np.array(existing[k].cpu().numpy()))
                if save_plots:
                    fig = plot_3d_point_cloud(existing[k][0], existing[k][1], existing[k][2], in_u_sphere=True, show=False)
                    fig.savefig(join(results_dir, 'fixed', f'{cat_name}_{i * batch_size + k}_existing.png'))
                    plt.close(fig)


def evaluate_generativity(full_model: FullModel, device, datasets_dict, results_dir, epoch, batch_size, num_workers,
                          mean=0.0, std=0.005):
    dataloaders_dict = {cat_name: DataLoader(cat_ds, pin_memory=True, batch_size=1, num_workers=num_workers)
                        for cat_name, cat_ds in datasets_dict.items()}
    chamfer_loss = ChamferLoss().to(device)
    with torch.no_grad():
        results = {}

        for cat_name, dl in dataloaders_dict.items():
            cat_gt = []
            for data in dl:
                _, missing, _, _ = data
                missing = missing.to(device)
                cat_gt.append(missing)
            cat_gt = torch.cat(cat_gt).contiguous()

            cat_results = {}

            for data in tqdm(dl, total=len(dl)):
                existing, _, _, _ = data
                existing = existing.to(device)

                obj_recs = []

                for j in range(len(cat_gt)):
                    fixed_noise = torch.zeros(1, full_model.get_noise_size()).normal_(mean=mean, std=std).to(device)
                    reconstruction = full_model(existing, None, [1, 2048, 3], epoch, device, noise=fixed_noise)

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


def compute_mmd_tmd_uhd(full_model: FullModel, device, dataset, results_dir, epoch, batch_size=64):
    from utils.evaluation.total_mutual_diff import process as tmd
    from utils.evaluation.completeness import process as uhd
    from utils.evaluation.mmd import process as mmd

    res = {}
    shape_dir_path = join(results_dir, 'fixed')

    mmd_v = mmd(shape_dir_path, dataset, device, batch_size)
    print('MMD * 1000', mmd_v * 1000)
    res['MMD * 1000'] = mmd_v * 1000

    uhd_v = uhd(shape_dir_path)
    print('UHD * 100', uhd_v * 100)
    res['UHD * 100'] = uhd_v * 100

    tmd_v = tmd(shape_dir_path)
    print('TMD * 100', tmd_v * 100)
    res['TMD * 100'] = tmd_v * 100

    with open(join(results_dir, 'compute_mmd_tmd_uhd', str(epoch) + 'res.json'), mode='w') as f:
        json.dump(res, f)


def merge_different_categories(full_model, device, dataset, results_dir, epoch, amount=10, first_cat='car',
                               second_cat='airplane'):
    first_cat_dataset = dataset[first_cat]
    second_cat_dataset = dataset[second_cat]

    if len(first_cat_dataset) < amount or len(second_cat_dataset) < amount:
        raise ValueError(f'with current dataset config the max amount value is '
                         f'{np.min([len(first_cat_dataset), len(second_cat_dataset)])}')

    first_cat_ids = np.random.choice(len(first_cat_dataset), amount, replace=False)
    second_cat_ids = np.random.choice(len(first_cat_dataset), amount, replace=False)

    with torch.no_grad():
        for i in range(amount):
            f_existing, f_missing, f_gt, _ = first_cat_dataset[first_cat_ids[i]]

            s_existing, s_missing, s_gt, _ = second_cat_dataset[second_cat_ids[i]]

            f_existing = f_gt[f_gt.T[0].argsort()[1024:]]
            f_missing = f_gt[f_gt.T[0].argsort()[:1024]]
            s_existing = s_gt[s_gt.T[0].argsort()[1024:]]
            s_missing = s_gt[s_gt.T[0].argsort()[:1024]]

            np.save(join(results_dir, 'merge_different_categories', f'{first_cat}_{i}_existing'), f_existing)
            np.save(join(results_dir, 'merge_different_categories', f'{first_cat}_{i}_missing'), f_missing)
            np.save(join(results_dir, 'merge_different_categories', f'{first_cat}_{i}_gt'), f_gt)

            np.save(join(results_dir, 'merge_different_categories', f'{second_cat}_{i}_existing'), s_existing)
            np.save(join(results_dir, 'merge_different_categories', f'{second_cat}_{i}_missing'), s_missing)
            np.save(join(results_dir, 'merge_different_categories', f'{second_cat}_{i}_gt'), s_gt)

            f_existing = torch.from_numpy(f_existing).unsqueeze(0).to(device)
            s_existing = torch.from_numpy(s_existing).unsqueeze(0).to(device)

            gt_shape = list(torch.from_numpy(f_gt).unsqueeze(0).shape)

            for j in range(amount):
                _, temp_f_missing, temp_f_gt, _ = first_cat_dataset[first_cat_ids[j]]
                _, temp_s_missing, temp_s_gt, _ = second_cat_dataset[second_cat_ids[j]]

                temp_f_missing = temp_f_gt[temp_f_gt.T[0].argsort()[:1024]]
                temp_s_missing = temp_s_gt[temp_s_gt.T[0].argsort()[:1024]]

                temp_f_missing = torch.from_numpy(temp_f_missing).unsqueeze(0).to(device)
                temp_s_missing = torch.from_numpy(temp_s_missing).unsqueeze(0).to(device)

                rec_ff = full_model(f_existing, temp_f_missing, gt_shape, epoch, device)
                np.save(join(results_dir, 'merge_different_categories', f'{first_cat}_{i}~{first_cat}_{j}_rec'),
                        rec_ff.cpu().numpy()[0].T)

                rec_fs = full_model(f_existing, temp_s_missing, gt_shape, epoch, device)
                np.save(join(results_dir, 'merge_different_categories', f'{first_cat}_{i}~{second_cat}_{j}_rec'),
                        rec_fs.cpu().numpy()[0].T)

                rec_sf = full_model(s_existing, temp_f_missing, gt_shape, epoch, device)
                np.save(join(results_dir, 'merge_different_categories', f'{second_cat}_{i}~{first_cat}_{j}_rec'),
                        rec_sf.cpu().numpy()[0].T)

                rec_ss = full_model(s_existing, temp_f_missing, gt_shape, epoch, device)
                np.save(join(results_dir, 'merge_different_categories', f'{second_cat}_{i}~{second_cat}_{j}_rec'),
                        rec_ss.cpu().numpy()[0].T)


def same_model_different_slices(full_model, device, datasets_dict, results_dir, epoch, amount=10, slices_number=10,
                                mean=0.0, std=0.015):
    def process_existing(pcd, cat_name, name, i, j):
        np.save(join(results_dir, 'same_model_different_slices', f'{cat_name}_{i}_{j}_{name}_pcd'), pcd)
        noise = torch.zeros(1, full_model.get_noise_size()).normal_(mean=mean, std=std)
        np.save(join(results_dir, 'same_model_different_slices', f'{cat_name}_{i}_{j}_{name}_noise'), noise.numpy())

        pcd = torch.from_numpy(pcd).unsqueeze(0).to(device)
        noise = noise.to(device)
        rec = full_model(pcd, None, [1, 2048, 3], epoch, device, noise=noise)[0].cpu().numpy()

        np.save(join(results_dir, 'same_model_different_slices', f'{cat_name}_{i}_{j}_{name}_rec'), rec)

        fig = plot_3d_point_cloud(rec[0], rec[1], rec[2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'same_model_different_slices', f'{cat_name}_{i}_{j}_{name}_rec.png'))
        plt.close(fig)

    with torch.no_grad():
        for cat_name, ds in datasets_dict.items():
            ids = np.random.choice(len(ds), amount, replace=False)
            for i, idx in tqdm(enumerate(ids), total=len(ids)):
                _, _, points, _ = ds[idx]
                points = points.T
                fig = plot_3d_point_cloud(points[0], points[1], points[2], in_u_sphere=True, show=False)
                fig.savefig(join(results_dir, 'same_model_different_slices', f'{cat_name}_{i}_gt.png'))
                plt.close(fig)
                points = points.T
                np.save(join(results_dir, 'same_model_different_slices', f'{cat_name}_{i}_gt'), points)
                for j in range(slices_number):
                    f_pcd, s_pcd = SlicedDatasetGenerator.generate_item(points, 1024)
                    process_existing(f_pcd, cat_name, 'f', i, j)
                    process_existing(s_pcd, cat_name, 's', i, j)


def completion3d_submission(full_model, device, datasets_dict, results_dir, epoch, batch_size=1):
    benchmark_dir = join(results_dir, 'completion3d_submission')

    shutil.rmtree(benchmark_dir, ignore_errors=True)
    os.makedirs(benchmark_dir, exist_ok=True)

    submission_zip = ZipFile('submission.zip', 'w')

    test_dataloader = DataLoader(datasets_dict['all'], batch_size=batch_size)

    for i, point_data in tqdm(enumerate(test_dataloader, 1), total=len(test_dataloader)):
        existing, _, _, model_id = point_data
        existing = existing.to(device)
        reconstruction = full_model(existing, None, [batch_size, 2048, 3], epoch, device).cpu()
        for idx, x in enumerate(reconstruction.permute(0, 2, 1)):
            ofile = join(benchmark_dir, model_id[idx].split('/')[-1] + '.h5')
            with h5py.File(ofile, "w") as f:
                f.create_dataset("data", data=x.numpy())
                f.close()
            submission_zip.write(ofile, 'all/' + basename(ofile))


def temp_exp(full_model, device, dataset_dict, results_dir, epoch):
    pass

    cat_name = 'car'
    amount = 100

    latent_tsne = np.load(join(results_dir, 'temp_exp', f'{cat_name}_latent_tsne.npy'))
    tnw_tsne = np.load(join(results_dir, 'temp_exp', f'{cat_name}_tnw_tsne.npy'))

    cat_test_tsne = latent_tsne[-(2 * amount):]
    cat_test_tnw = tnw_tsne[-(2 * amount):]

    latent_dist = np.zeros(amount)
    tnw_dist = np.zeros(amount)

    for i in range(amount):
        latent_dist[i] = np.linalg.norm(cat_test_tsne[2 * i] - cat_test_tsne[2 * i + 1])
        tnw_dist[i] = np.linalg.norm(cat_test_tnw[2 * i] - cat_test_tnw[2 * i + 1])

    plt.plot(latent_tsne.T[0], latent_tsne.T[1], 'o', cat_test_tsne.T[0], cat_test_tsne.T[1], 'o')
    plt.title('latent')
    plt.show()

    plt.plot(tnw_tsne.T[0], tnw_tsne.T[1], 'o', cat_test_tnw.T[0][0], cat_test_tnw.T[1][0], 'o', cat_test_tnw.T[0][1],
             cat_test_tnw.T[1][1], 'o')
    plt.title('tnw')
    plt.show()

    # np.save(join(results_dir, 'temp_exp', f'{cat_name}_latent_tsne_dist'), latent_dist)
    # np.save(join(results_dir, 'temp_exp', f'{cat_name}_tnw_tsne_dist'), tnw_dist)

    exit(0)

    from datasets.shapenet import ShapeNetDataset

    train_dataset_dict = ShapeNetDataset._get_datasets_for_classes(
        'D:\\UJ\\bachelors\\3d-point-clouds-autocomplete\\data\\shapenet',
        'train',
        use_pcn_model_list=True,
        is_random_rotated=False,
        num_samples=1,
        # classes=['04530566', '02933112']
    )

    is_compute = False

    with torch.no_grad():

        latents = {}
        tnws = {}

        if is_compute:
            dataloaders_dict = {cat_name: DataLoader(cat_ds, pin_memory=True, batch_size=1, num_workers=0)
                                for cat_name, cat_ds in train_dataset_dict.items()}
            for cat_name, dl in dataloaders_dict.items():

                if cat_name != 'car':
                    continue

                cat_latent = []
                cat_tnw = []

                for data in tqdm(dl, total=len(dl)):
                    existing, missing, gt, _ = data
                    existing = existing.to(device)
                    missing = missing.to(device)

                    rec, latent, tnw = full_model(existing, missing, list(gt.shape), epoch, device)

                    cat_latent.append(latent.detach().cpu())
                    cat_tnw.append(tnw.detach().cpu())

                latents[cat_name] = torch.cat(cat_latent).numpy()
                tnws[cat_name] = torch.cat(cat_tnw).numpy()

            latents['all'] = np.concatenate([v for v in latents.values()])
            tnws['all'] = np.concatenate([v for v in tnws.values()])

            for cat_name in latents.keys():
                np.save(join(results_dir, 'temp_exp', f'{cat_name}_latent1'), latents[cat_name])
                np.save(join(results_dir, 'temp_exp', f'{cat_name}_tnw1'), tnws[cat_name])
        else:
            for cat_name in train_dataset_dict.keys():
                if cat_name != 'car':
                    continue
                latents[cat_name] = np.load(join(results_dir, 'temp_exp', f'{cat_name}_latent1.npy'))
                tnws[cat_name] = np.load(join(results_dir, 'temp_exp', f'{cat_name}_tnw1.npy'))

        for cat_name, ds in dataset_dict.items():

            if cat_name != 'car':
                continue

            cat_ids = np.random.choice(len(ds), amount, replace=False)

            cat_latent = []
            cat_tnw = []

            for i in range(amount):
                _, _, gt, _ = ds[cat_ids[i]]

                np.save(join(results_dir, 'temp_exp', 'gts', f'{cat_name}_{i}'), gt)

                existing_x = gt[gt.T[0].argsort()[1024:]]
                missing_x = gt[gt.T[0].argsort()[:1024]]

                existing_y = gt[gt.T[1].argsort()[1024:]]
                missing_y = gt[gt.T[1].argsort()[:1024]]

                gt_shape = list(torch.from_numpy(gt).unsqueeze(0).shape)

                existing = torch.from_numpy(existing_x).unsqueeze(0).to(device)
                missing = torch.from_numpy(missing_x).unsqueeze(0).to(device)
                _, latent, tnw = full_model(existing, missing, gt_shape, epoch, device)
                cat_latent.append(latent.cpu())
                cat_tnw.append(tnw.cpu())

                existing = torch.from_numpy(existing_y).unsqueeze(0).to(device)
                missing = torch.from_numpy(missing_y).unsqueeze(0).to(device)
                _, latent, tnw = full_model(existing, missing, gt_shape, epoch, device)
                cat_latent.append(latent.cpu())
                cat_tnw.append(tnw.cpu())

            cat_latent = torch.cat(cat_latent).numpy()
            cat_tnw = torch.cat(cat_tnw).numpy()

            cc_latent = np.concatenate([latents[cat_name], cat_latent])
            cc_tnw = np.concatenate([tnws[cat_name], cat_tnw])

            start_time = datetime.now()
            print(start_time)
            latent_tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(cc_latent)
            print(datetime.now() - start_time)
            cat_test_tsne = latent_tsne[-(2 * amount):]
            plt.plot(latent_tsne.T[0], latent_tsne.T[1], 'o', cat_test_tsne.T[0], cat_test_tsne.T[1], 'o')
            plt.title('latent')
            plt.show()

            np.save(join(results_dir, 'temp_exp', f'{cat_name}_latent_tsne'), latent_tsne)

            start_time = datetime.now()
            print(start_time)
            tnw_tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(cc_tnw)
            print(datetime.now() - start_time)
            cat_test_tnw = tnw_tsne[-(2 * amount):]
            plt.plot(tnw_tsne.T[0], tnw_tsne.T[1], 'o', cat_test_tnw.T[0], cat_test_tnw.T[1], 'o')
            plt.title('tnw')
            plt.show()

            np.save(join(results_dir, 'temp_exp', f'{cat_name}_tnw_tsne'), tnw_tsne)

        '''
        
        latent_tsne = {cat: manifold.TSNE(n_components=2, init='pca').fit_transform(latent) for cat, latent in latents.items()}

        for c, l in latent_tsne.items():
            plt.plot(l.T[0], l.T[1], 'o')
            plt.title(c)
            plt.show()
        pass
        '''


experiment_functions_dict = {
    'fixed': fixed,
    'evaluate_generativity': evaluate_generativity,
    'compute_mmd_tmd_uhd': compute_mmd_tmd_uhd,
    'merge_different_categories': merge_different_categories,
    'same_model_different_slices': same_model_different_slices,
    "completion3d_submission": completion3d_submission,
    "temp_exp": temp_exp
}
