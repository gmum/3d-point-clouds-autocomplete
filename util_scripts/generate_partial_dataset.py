import csv
import os
from os.path import join, exists

import ray
import trimesh
import numpy as np

from core.arg_parser import parse_config
from datasets.shapenet_3depn import sample_point_cloud_by_n
from datasets.utils.dataset_generator import SlicedDatasetGenerator
from datasets.utils.shapenet_category_mapping import synth_id_to_category
from utils.plyfile import quick_save_ply_file, load_ply
from utils.util import get_filenames_by_cat


@ray.remote
def generate_one_shapenet(category: str, filename: str, dataset_path: str, num_samples: int = 4):
    pc_filepath = join(dataset_path, category, filename)
    points = load_ply(pc_filepath)

    for i in range(num_samples):
        existing, missing = SlicedDatasetGenerator.generate_item(points)
        quick_save_ply_file(existing, join(dataset_path, 'slices', 'existing', category, str(i) + '~' + filename))
        quick_save_ply_file(missing, join(dataset_path, 'slices', 'missing', category, str(i) + '~' + filename))


@ray.remote
def generate_one_3depn(cat: str, name: str, dataset_path: str, pc_root: str, num_samples: int = 4):
    ply_path = join(pc_root, name + '.ply')

    pc = np.array(trimesh.load(ply_path).vertices)
    pc = sample_point_cloud_by_n(pc, 2048)

    quick_save_ply_file(pc, join(dataset_path, 'slices', 'gt', cat, name + '.ply'))

    for i in range(num_samples):
        existing, missing = SlicedDatasetGenerator.generate_item(pc)
        quick_save_ply_file(existing, join(dataset_path, 'slices', 'existing', cat, str(i) + '~' + name + '.ply'))
        quick_save_ply_file(missing, join(dataset_path, 'slices', 'missing', cat, str(i) + '~' + name + '.ply'))


def main(config: dict):
    dataset_config: dict = config['dataset']

    dataset_path: str = dataset_config['path']
    dataset_name: str = dataset_config['name']
    num_samples: int = dataset_config['num_samples']

    if dataset_name == 'shapenet':
        if not exists(join(dataset_path)):
            raise Exception(f'no ShapeNet dataset found at {dataset_path}, '
                            f'please run `util_scripts/download_shapenet_2048.py` first')

        for category in synth_id_to_category.keys():
            os.makedirs(join(dataset_path, 'slices', 'existing', category), exist_ok=True)
            os.makedirs(join(dataset_path, 'slices', 'missing', category), exist_ok=True)

        ray.init(num_cpus=os.cpu_count())
        ray.get([generate_one_shapenet.remote(row['category'], row['filename'], dataset_path, num_samples) for _, row in
                 get_filenames_by_cat(dataset_path).iterrows()])
        ray.shutdown()

    elif dataset_name == '3depn':
        classes: list = ['02691156', '03001627', '04379243']

        cat_pc_root: dict = {cat: join(dataset_path, 'ShapeNetPointCloud', cat) for cat in classes}
        cat_pc_raw_root: dict = {cat: join(dataset_path, 'shapenet_dim32_sdf_pc', cat) for cat in classes}
        cat_shape_names: dict = {cat: [] for cat in classes}

        with open(join(dataset_path, 'shapenet-official-split.csv'), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_cnt = 0
            for row in csv_reader:
                if line_cnt == 0 or (row[1] not in classes):
                    pass
                else:
                    if row[-1] in ['train', 'val']:
                        cat_shape_names[row[1]].append(row[-2])
                line_cnt += 1

        refined_shape_names: dict = {cat: [] for cat in classes}
        for cat, shapes in cat_shape_names.items():
            for name in shapes:
                ply_path = join(cat_pc_root[cat], name + '.ply')
                path = join(cat_pc_raw_root[cat], f'{name}__0__.ply')
                if exists(ply_path) and exists(path):
                    refined_shape_names[cat].append(name)

        for cat in classes:
            os.makedirs(join(dataset_path, 'slices', 'existing', cat), exist_ok=True)
            os.makedirs(join(dataset_path, 'slices', 'missing', cat), exist_ok=True)
            os.makedirs(join(dataset_path, 'slices', 'gt', cat), exist_ok=True)

        print('pc to process: ', np.sum([len(v) for v in refined_shape_names.values()]))
        print('pc to process: ', {k: len(v) for k, v in refined_shape_names.items()})

        ray.init(num_cpus=os.cpu_count())

        ray_tasks = []
        for cat, shapes in refined_shape_names.items():
            for name in shapes:
                ray_tasks.append(generate_one_3depn.remote(cat, name, dataset_path, cat_pc_root[cat], num_samples))
                # single thread version
                # ply_path = join(cat_pc_root[cat], name + '.ply')
                #
                # pc = np.array(trimesh.load(ply_path).vertices)
                # pc = sample_point_cloud_by_n(pc, 2048)
                #
                # quick_save_ply_file(pc, join(dataset_path, 'slices', 'existing', cat, str(i) + '~' + name + '.ply'))
                #
                # for i in range(4):
                #     existing, missing = SlicedDatasetGenerator.generate_item(pc)
                #     quick_save_ply_file(existing, join(dataset_path, 'slices', 'existing', cat,
                #                                       str(i) + '~' + name + '.ply'))
                #     quick_save_ply_file(missing, join(dataset_path, 'slices', 'missing', cat,
                #                                         str(i) + '~' + name + '.ply'))
        ray.get(ray_tasks)
        ray.shutdown()


if __name__ == '__main__':
    main(parse_config())
