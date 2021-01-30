import numpy as np
import ray
import trimesh

import csv
import random
import os
from os.path import join, exists

from core.arg_parser import parse_config
from datasets.shapenet_3depn import sample_point_cloud_by_n
from datasets.utils.dataset_generator import SlicedDatasetGenerator
from utils.plyfile import quick_save_ply_file


@ray.remote
def generate_one(cat, name, dataset_path, pc_root):
    ply_path = join(pc_root, name + '.ply')

    pc = np.array(trimesh.load(ply_path).vertices)
    pc = sample_point_cloud_by_n(pc, 2048)

    quick_save_ply_file(pc, join(dataset_path, 'slices', 'gt', cat,
                                 name + '.ply'))

    for i in range(4):
        partial, remaining = SlicedDatasetGenerator.generate_item(pc)
        quick_save_ply_file(partial, join(dataset_path, 'slices', 'partial', cat,
                                          str(i) + '~' + name + '.ply'))
        quick_save_ply_file(remaining, join(dataset_path, 'slices', 'remaining', cat,
                                            str(i) + '~' + name + '.ply'))


def main(config: dict):
    dataset_config: dict = config['dataset']

    dataset_path: str = dataset_config['path']
    dataset_name: str = dataset_config['name']

    if dataset_name == 'shapenet':
        pass
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
            os.makedirs(join(dataset_path, 'slices', 'partial', cat), exist_ok=True)
            os.makedirs(join(dataset_path, 'slices', 'remaining', cat), exist_ok=True)
            os.makedirs(join(dataset_path, 'slices', 'gt', cat), exist_ok=True)

        print('pc to process: ', np.sum([len(v) for v in refined_shape_names.values()]))
        print('pc to process: ', {k: len(v) for k, v in refined_shape_names.items()})

        ray.init(num_cpus=os.cpu_count())

        ray_tasks = []
        for cat, shapes in refined_shape_names.items():
            for name in shapes:
                ray_tasks.append(generate_one.remote(cat, name, dataset_path, cat_pc_root[cat]))
                '''
                ply_path = join(cat_pc_root[cat], name + '.ply')

                pc = np.array(trimesh.load(ply_path).vertices)
                pc = sample_point_cloud_by_n(pc, 2048)

                quick_save_ply_file(pc, join(dataset_path, 'slices', 'partial', cat,
                                             str(i) + '~' + name + '.ply'))
                
                for i in range(4):
                    partial, remaining = SlicedDatasetGenerator.generate_item(pc)
                    quick_save_ply_file(partial, join(dataset_path, 'slices', 'partial', cat, 
                                                      str(i) + '~' + name + '.ply'))
                    quick_save_ply_file(remaining, join(dataset_path, 'slices', 'remaining', cat,
                                                        str(i) + '~' + name + '.ply'))
                '''
        ray.get(ray_tasks)
        ray.shutdown()


if __name__ == '__main__':
    main(parse_config())
