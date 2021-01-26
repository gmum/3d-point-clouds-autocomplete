import numpy as np
import trimesh
import torch

import csv
import os
import random
from os.path import join, exists

from datasets.base_dataset import BaseDataset
from datasets.utils.shapenet_category_mapping import synth_id_to_category


def downsample_point_cloud(points, n_pts):
    """downsample points by random choice

    :param points: (n, 3)
    :param n_pts: int
    :return:
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts)
    return points[p_idx]


def upsample_point_cloud(points, n_pts):
    """upsample points by random choice

    :param points: (n, 3)
    :param n_pts: int, > n
    :return:
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts - points.shape[0])
    dup_points = points[p_idx]
    points = np.concatenate([points, dup_points], axis=0)
    return points


def sample_point_cloud_by_n(points, n_pts):
    """resample point cloud to given number of points"""
    if n_pts > points.shape[0]:
        return upsample_point_cloud(points, n_pts)
    elif n_pts < points.shape[0]:
        return downsample_point_cloud(points, n_pts)
    else:
        return points


def collect_train_split_by_id(path, cat_id):
    split_info = {"train":[], 'validation':[], 'test':[]}
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_cnt = 0
        for row in csv_reader:
            if line_cnt == 0 or row[1] != cat_id:
                pass
            else:
                if row[-1] == "train":
                    split_info["train"].append(row[-2])
                elif row[-1] == "val":
                    split_info["validation"].append(row[-2])
                else:
                    split_info["test"].append(row[-2])
            line_cnt += 1
    return split_info


class ShapeNet3DEPNDataset(BaseDataset):

    def __init__(self, root_dir='/home/datasets/completion', split='train', classes=[]):
        super(ShapeNet3DEPNDataset, self).__init__(root_dir, split, classes)

        self.cat_pc_root = join(root_dir, 'ShapeNetPointCloud', classes[0])
        self.cat_pc_raw_root = join(root_dir, 'shapenet_dim32_sdf_pc', classes[0])

        shape_names = []

        with open(join(self.root_dir, 'shapenet-official-split.csv'), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_cnt = 0
            for row in csv_reader:
                if line_cnt == 0 or (row[1] not in classes if classes else False):
                    pass
                else:
                    if row[-1] == self.split:
                        shape_names.append(row[-2])
                line_cnt += 1

        self.shape_names = []
        for name in shape_names:
            ply_path = join(self.cat_pc_root, name + '.ply')
            path = join(self.cat_pc_raw_root, "{}__0__.ply".format(name))
            if exists(ply_path) and exists(path):
                self.shape_names.append(name)

        self.raw_ply_names = sorted(os.listdir(self.cat_pc_raw_root))

        self.rng = random.Random(1234)  # from original publication

    def __getitem__(self, index):
        if self.split == "train":
            raw_n = random.randint(0, len(self.raw_ply_names) - 1)
            raw_pc_name = self.raw_ply_names[raw_n]
        else:
            raw_n = self.rng.randint(0, 7)
            raw_pc_name = self.shape_names[index] + "__{}__.ply".format(raw_n)

        # process partial shapes
        raw_ply_path = os.path.join(self.cat_pc_raw_root, raw_pc_name)
        raw_pc = np.array(trimesh.load(raw_ply_path).vertices)
        raw_pc = self._rotate_point_cloud_by_axis_angle(raw_pc)
        raw_pc = sample_point_cloud_by_n(raw_pc, 1024)
        raw_pc = torch.tensor(raw_pc, dtype=torch.float32).transpose(1, 0)

        # process real complete shapes
        real_shape_name = self.shape_names[index]
        real_ply_path = os.path.join(self.cat_pc_root, real_shape_name + '.ply')
        real_pc = np.array(trimesh.load(real_ply_path).vertices)
        real_pc = sample_point_cloud_by_n(real_pc, 2048)
        real_pc = torch.tensor(real_pc, dtype=torch.float32).transpose(1, 0)

        # partial, remaining, gt, model_id
        return raw_pc, real_pc, real_pc, raw_pc_name + '~' + real_shape_name

        # return {"raw": raw_pc, "real": real_pc, "raw_id": raw_pc_name, "real_id": real_shape_name}

    def __len__(self):
        return len(self.shape_names)

    def _rotate_point_cloud_by_axis_angle(self, points):
        rot_m = np.array([[2.22044605e-16, 0.00000000e+00, 1.00000000e+00],
                          [0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                          [-1.00000000e+00, 0.00000000e+00, 2.22044605e-16]])

        return np.dot(rot_m, points.T).T

    @classmethod
    def get_validation_datasets(cls, root_dir, classes=[], **kwargs):
        if not classes:
            classes = ['02691156', '03001627', '04379243']

        return {synth_id_to_category[category_id]: ShapeNet3DEPNDataset(root_dir=root_dir, split='val',
                                                                        classes=[category_id])
                for category_id in classes}
