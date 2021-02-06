from os import makedirs
from os.path import join, exists
import numpy as np
from tqdm import tqdm

from core.arg_parser import parse_config
from datasets.utils.dataset_generator import HyperPlane
from utils.plyfile import load_ply, quick_save_ply_file


def div_left_right_bin_search(dataset_dir, init_plane_points, pc_paths):
    for i, pc_path in tqdm(enumerate(pc_paths), total=len(pc_paths)):

        pc = load_ply(join(dataset_dir, pc_path))

        points = init_plane_points.copy()

        l, r = pc.T[1].min(), pc.T[1].max()

        counter = 0

        while True:

            m = np.divide(l + r, 2)

            points[0][1] = m
            points[1][1] = m
            points[2][1] = m

            right = HyperPlane.get_plane_from_3_points(points).check_point(pc) > 0
            right_points = pc[right]
            left_points = pc[~right]

            counter += 1
            if counter == 100000000:
                quick_save_ply_file(right_points, join(dataset_dir, 'test_gen', 'right', pc_path))
                quick_save_ply_file(left_points, join(dataset_dir, 'test_gen', 'left', pc_path))
                quick_save_ply_file(pc, join(dataset_dir, 'test_gen', 'gt', pc_path))
                break

            if len(left_points) > len(right_points):
                l = m
            elif len(left_points) < len(right_points):
                r = m
            else:
                quick_save_ply_file(left_points, join(dataset_dir, 'test_gen', 'left', pc_path))
                quick_save_ply_file(right_points, join(dataset_dir, 'test_gen', 'right', pc_path))
                quick_save_ply_file(pc, join(dataset_dir, 'test_gen', 'gt', pc_path))
                break


def div_left_right_min_y(dataset_dir, pc_paths):
    for i, pc_path in tqdm(enumerate(pc_paths), total=len(pc_paths)):
        pc = load_ply(join(dataset_dir, pc_path))

        right_points = pc[pc.T[1].argsort()[1024:]]
        left_points = pc[pc.T[1].argsort()[:1024]]

        quick_save_ply_file(left_points, join(dataset_dir, 'test_gen', 'left', pc_path))
        quick_save_ply_file(right_points, join(dataset_dir, 'test_gen', 'right', pc_path))
        quick_save_ply_file(pc, join(dataset_dir, 'test_gen', 'gt', pc_path))


def main(config):
    dataset_dir = config['dataset']['path']

    with open(join(dataset_dir, 'test.list')) as file:
        pc_paths = [line.strip() + '.ply' for line in file]

    plane_points = np.zeros((3, 3))
    plane_points[1][2] = 1
    plane_points[2][0] = 1

    for cat in ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']:
        makedirs(join(dataset_dir, 'test_gen', 'left', cat), exist_ok=True)
        makedirs(join(dataset_dir, 'test_gen', 'right', cat), exist_ok=True)
        makedirs(join(dataset_dir, 'test_gen', 'gt', cat), exist_ok=True)

    div_left_right_min_y(dataset_dir, pc_paths)

    not_existed_pc = []

    for pc_path in pc_paths:
        if not (exists(join(dataset_dir, 'test_gen', 'left', pc_path))
                and exists(join(dataset_dir, 'test_gen', 'left', pc_path))):
            not_existed_pc.append(pc_path)

    # div_left_right_bin_search(dataset_dir, plane_points, not_existed_pc)

    not_1024 = []
    for pc_path in pc_paths:
        if load_ply(join(dataset_dir, 'test_gen', 'left', pc_path)).shape[0] != 1024:
            not_1024.append(pc_path)


if __name__ == '__main__':
    main(parse_config())
