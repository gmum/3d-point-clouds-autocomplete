import argparse
import os
import numpy as np

from tqdm import tqdm
import ray

from utils.evaluation.chamfer import compute_trimesh_chamfer
import glob
from joblib import Parallel, delayed


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

    # pc_paths = glob.glob(os.path.join(shape_dir, "fake-z*.ply"))

    pc_paths = sorted(pc_paths)
    gen_pcs = []

    for i in range(int(len(pc_paths)/10)):
        pcs = []
        for j in range(10):
            pcs.append(np.load(pc_paths[i*10+j]).T)
        gen_pcs.append(pcs)
    gen_pcs = np.array(gen_pcs)

    # ray.init(num_cpus=os.cpu_count())
    # ray_tmd_tasks = [process_one_tmd.remote(gen_pcs[i]) for i in range(len(gen_pcs))]
    # tmd = ray.get(ray_tmd_tasks)
    # ray.shutdown()
    # return np.mean(tmd)

    results = []
    for i in tqdm(range(len(gen_pcs))):
        sum_dist = 0
        for j in range(len(gen_pcs[i])):
            for k in range(j + 1, len(gen_pcs[i]), 1):
                pc1 = gen_pcs[i][j]
                pc2 = gen_pcs[i][k]
                chamfer_dist = compute_trimesh_chamfer(pc1, pc2)
                sum_dist += chamfer_dist
        mean_dist = sum_dist * 2 / (len(gen_pcs[i]) - 1)
        results.append(mean_dist)

    return np.mean(results)


def process_one(shape_dir):
    pc_paths = glob.glob(os.path.join(shape_dir, "*reconstruction.npy"))

    #pc_paths = glob.glob(os.path.join(shape_dir, "fake-z*.ply"))

    pc_paths = sorted(pc_paths)
    gen_pcs = []
    for path in pc_paths:
        gen_pcs.append(np.load(path).T)
        #sample_pts = trimesh.load(path)
        #sample_pts = sample_pts.vertices
        #gen_pcs.append(sample_pts)

    sum_dist = 0
    for j in range(len(gen_pcs)):
        for k in range(j + 1, len(gen_pcs), 1):
            pc1 = gen_pcs[j]
            pc2 = gen_pcs[k]
            chamfer_dist = compute_trimesh_chamfer(pc1, pc2)
            sum_dist += chamfer_dist
    mean_dist = sum_dist * 2 / (len(gen_pcs) - 1)
    return mean_dist


def Total_Mutual_Difference(args):
    shape_names = sorted(os.listdir(args.src))
    res = 0
    all_shape_dir = [os.path.join(args.src, name) for name in shape_names]

    results = Parallel(n_jobs=args.process, verbose=2)(delayed(process_one)(path) for path in all_shape_dir)

    info_path = args.src + '-record_meandist.txt'
    with open(info_path, 'w') as fp:
        for i in range(len(shape_names)):
            print("ID: {} \t mean_dist: {:.4f}".format(shape_names[i], results[i]), file=fp)
    res = np.mean(results)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("-p", "--process", type=int, default=10)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.src + '-eval_TMD.txt'

    res = Total_Mutual_Difference(args)
    print("Avg Total Multual Difference: {}".format(res))

    with open(args.output, "w") as fp:
        fp.write("SRC: {}\n".format(args.src))
        fp.write("Total Multual Difference: {}\n".format(res))


if __name__ == '__main__':
    main()