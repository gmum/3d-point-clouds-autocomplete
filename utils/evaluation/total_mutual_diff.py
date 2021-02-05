import os
import glob

import ray
import numpy as np
from tqdm import tqdm

from utils.evaluation.chamfer import compute_trimesh_chamfer

#  code is based on
#  https://github.com/ChrisWu1997/Multimodal-Shape-Completion/blob/master/evaluation/total_mutual_diff.py


@ray.remote
def process_one_tmd(gen_pcs):
    sum_dist = 0
    for j in range(len(gen_pcs)):
        for k in range(j + 1, len(gen_pcs), 1):
            pc1 = gen_pcs[j]
            pc2 = gen_pcs[k]
            chamfer_dist = compute_trimesh_chamfer(pc1, pc2)
            sum_dist += chamfer_dist
    mean_dist = sum_dist * 2 / (len(gen_pcs) - 1)
    return mean_dist


def process(shape_dir):
    pc_paths = glob.glob(os.path.join(shape_dir, "*reconstruction.npy"))

    pc_paths = sorted(pc_paths)
    gen_pcs = []

    for i in range(int(len(pc_paths)/10)):
        pcs = []
        for j in range(10):
            pcs.append(np.load(pc_paths[i*10+j]).T)
        gen_pcs.append(pcs)
    gen_pcs = np.array(gen_pcs)

    # parallel version
    # ray.init(num_cpus=os.cpu_count())
    # ray_tmd_tasks = [process_one_tmd.remote(gen_pcs[i]) for i in range(len(gen_pcs))]
    # tmd = ray.get(ray_tmd_tasks)
    # ray.shutdown()
    # return np.mean(tmd)

    results = []
    pbar = tqdm(range(len(gen_pcs)))
    for i in pbar:
        sum_dist = 0
        for j in range(len(gen_pcs[i])):
            for k in range(j + 1, len(gen_pcs[i]), 1):
                pc1 = gen_pcs[i][j]
                pc2 = gen_pcs[i][k]
                chamfer_dist = compute_trimesh_chamfer(pc1, pc2)
                sum_dist += chamfer_dist
        mean_dist = sum_dist * 2 / (len(gen_pcs[i]) - 1)
        results.append(mean_dist)
        pbar.set_postfix({"mmd": np.mean(results)})

    return np.mean(results)
