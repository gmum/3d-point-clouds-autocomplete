import os
import glob
import warnings

import ray
import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree

#  code is based on
#  https://github.com/ChrisWu1997/Multimodal-Shape-Completion/blob/master/evaluation/completeness.py


def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """

    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist


def nn_distance(query_points, ref_points):
    ref_points_kd_tree = KDTree(ref_points)
    one_distances, one_vertex_ids = ref_points_kd_tree.query(query_points)
    return one_distances


def completeness(query_points, ref_points, thres=0.03):
    a2b_nn_distance = nn_distance(query_points, ref_points)
    percentage = np.sum(a2b_nn_distance < thres) / len(a2b_nn_distance)
    return percentage


@ray.remote
def process_one_uhd(partial, gen_pcs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen_pcs_tensors = [torch.tensor(pc) for pc in gen_pcs]
        gen_pcs_tensors = torch.stack(gen_pcs_tensors, dim=0)
        partial_pc_tensor = torch.tensor(partial).unsqueeze(0).repeat((gen_pcs_tensors.size(0), 1, 1))
        return directed_hausdorff(partial_pc_tensor, gen_pcs_tensors, reduce_mean=True).item()


def process(shape_dir):
    # load generated shape
    pc_paths = glob.glob(os.path.join(shape_dir, "*reconstruction.npy"))
    pc_paths = sorted(pc_paths)

    # load partial input
    partial_paths = glob.glob(os.path.join(shape_dir, "*partial.npy"))
    partial_paths = sorted(partial_paths)

    gen_pcs = []
    for i in range(int(len(pc_paths) / 10)):
        pcs = []
        for j in range(10):
            pcs.append(np.load(pc_paths[i * 10 + j]))
        gen_pcs.append(pcs)
    gen_pcs = np.array(gen_pcs)

    partial_pcs = []
    for i in range(len(partial_paths)):
        partial_pcs.append(np.load(partial_paths[i]))
    partial_pcs = np.array(partial_pcs)

    ray.init(num_cpus=4)
    ray_uhd_tasks = [process_one_uhd.remote(partial_pcs[i], gen_pcs[i]) for i in range(int(len(pc_paths) / 10))]
    uhd = np.mean(ray.get(ray_uhd_tasks))
    ray.shutdown()
    return uhd

    # single thread version
    #
    # # completeness percentage
    # gen_comp_res = []
    #
    # for i in range(len(gen_pcs)):
    #     gen_comp = 0
    #     for sample_pts in gen_pcs[i]:
    #         comp = completeness(partial_pcs[i], sample_pts)
    #         gen_comp += comp
    #     gen_comp_res.append(gen_comp / len(gen_pcs))
    #
    #
    # # unidirectional hausdorff
    # hausdorff_res = []
    #
    # for i in tqdm(range(len(gen_pcs))):
    #
    #     gen_pcs_tensors = [torch.tensor(pc).transpose(1, 0) for pc in gen_pcs[i]]
    #     gen_pcs_tensors = torch.stack(gen_pcs_tensors, dim=0)
    #
    #     partial_pc_tensor = torch.tensor(partial_pcs[i]).transpose(1, 0)
    #
    #     partial_pc_tensor = partial_pc_tensor.unsqueeze(0).repeat((gen_pcs_tensors.size(0), 1, 1))
    #
    #     hausdorff = directed_hausdorff(partial_pc_tensor, gen_pcs_tensors, reduce_mean=True).item()
    #     hausdorff_res.append(hausdorff)
    #
    # return np.mean(hausdorff_res) #  np.mean(gen_comp_res), np.mean(hausdorff_res)
    #
