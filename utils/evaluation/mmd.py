import numpy as np
from tqdm import tqdm

import random
import glob
import torch
from os.path import join

from utils.pytorch_structural_losses.nn_distance import nn_distance


def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]


def minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, device=None):

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')

    matched_dists = []
    pbar = tqdm(range(n_ref))
    for i in pbar:
        best_in_all_batches = []
        ref = torch.from_numpy(ref_pcs[i]).unsqueeze(0).to(device).contiguous()
        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            chunk = torch.from_numpy(sample_chunk).to(device).contiguous()
            ref_to_s, s_to_ref = nn_distance(ref, chunk)
            all_dist_in_batch = ref_to_s.mean(dim=1) + s_to_ref.mean(dim=1)
            best_in_batch = torch.min(all_dist_in_batch).item()
            best_in_all_batches.append(best_in_batch)

        matched_dists.append(np.min(best_in_all_batches))
        pbar.set_postfix({"mmd": np.mean(matched_dists)})

    mmd = np.mean(matched_dists)
    return mmd, matched_dists


def process(shape_dir, dataset, device, batch_size=64):
    random.seed(1234)
    ref_pcs = []
    for data in dataset:
        _, _, gt, _ = data
        ref_pcs.append(gt)
    ref_pcs = np.stack(ref_pcs, axis=0)

    pc_paths = glob.glob(join(shape_dir, "*reconstruction.npy"))
    pc_paths = sorted(pc_paths)

    sample_pcs = []
    for path in pc_paths:
        sample_pcs.append(np.load(path).T)
    sample_pcs = np.stack(sample_pcs, axis=0)

    mmd, matched_dists = minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, device)
    del sample_pcs
    del ref_pcs
    return mmd
