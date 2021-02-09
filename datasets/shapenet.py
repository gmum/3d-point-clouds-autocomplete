import os
from os import listdir
from os.path import exists, join

import numpy as np
import pandas as pd

from datasets.base_dataset import BaseDataset
from datasets.utils.shapenet_category_mapping import synth_id_to_category, category_to_synth_id, synth_id_to_number
from utils.plyfile import load_ply
from datasets.utils.dataset_generator import SlicedDatasetGenerator
from utils.util import resample_pcd, get_filenames_by_cat


class ShapeNetDataset(BaseDataset):

    def __init__(self, root_dir='/home/datasets/shapenet', split='train', classes=[],
                 is_random_rotated=False, num_samples=4, use_list_with_name='pcn', is_gen=False):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
        """
        super().__init__(root_dir, split, classes)

        self.is_random_rotated = is_random_rotated
        self.use_list_with_name = use_list_with_name
        self.num_samples = num_samples
        self.is_gen = is_gen

        if self.use_list_with_name is not None:

            if self.use_list_with_name == 'pcn':
                list_path = join(root_dir, self.split +'.list')
            elif self.use_list_with_name == 'msc':
                list_path = join(root_dir, self.split + '_msc.list')
            else:
                raise ValueError('use_list_with_name can have only values `pcn` or `msc`')

            with open(list_path) as file:
                if classes:
                    self.point_clouds_names = [line.strip() for line in file if line.strip().split('/')[0] in classes]
                else:
                    self.point_clouds_names = [line.strip() for line in file]
        else:
            pc_df = get_filenames_by_cat(self.root_dir)
            if classes:
                if classes[0] not in synth_id_to_category.keys():
                    classes = [category_to_synth_id[c] for c in classes]
                pc_df = pc_df[pc_df.category.isin(classes)].reset_index(drop=True)
            else:
                classes = synth_id_to_category.keys()

            if self.split == 'train':
                # first 85%
                self.point_clouds_names = pd.concat(
                    [pc_df[pc_df['category'] == c][:int(0.85 * len(pc_df[pc_df['category'] == c]))]
                         .reset_index(drop=True) for c in classes])
            elif self.split == 'val':
                # missing 5%
                self.point_clouds_names = pd.concat([pc_df[pc_df['category'] == c][
                                                     int(0.85 * len(pc_df[pc_df['category'] == c])):int(
                                                         0.9 * len(pc_df[pc_df['category'] == c]))]
                                                    .reset_index(drop=True) for c in classes])
            else:
                # last 10%
                self.point_clouds_names = pd.concat(
                    [pc_df[pc_df['category'] == c][int(0.9 * len(pc_df[pc_df['category'] == c])):]
                         .reset_index(drop=True) for c in classes])

    def __len__(self):
        return len(self.point_clouds_names) * self.num_samples

    def __getitem__(self, idx):

        if self.use_list_with_name is not None:
            pc_category, pc_filename = self.point_clouds_names[idx // self.num_samples].split('/')
            pc_filename += '.ply'
        else:
            pc_category, pc_filename = self.point_clouds_names.iloc[idx // self.num_samples].values

        if self.is_random_rotated:
            from scipy.spatial.transform import Rotation
            random_rotation_matrix = Rotation.from_euler('z', np.random.randint(360), degrees=True).as_matrix().astype(
                np.float32)

        scan_idx = str(idx % self.num_samples)

        if self.is_gen:
            # TODO ensure it is test set
            existing = resample_pcd(load_ply(join(self.root_dir, 'test_gen', 'right', pc_category, pc_filename)), 1024)
            missing = resample_pcd(load_ply(join(self.root_dir, 'test_gen', 'left', pc_category, pc_filename)), 1024)
            gt = load_ply(join(self.root_dir, 'test_gen', 'gt', pc_category, pc_filename))
        else:
            existing = load_ply(join(self.root_dir, 'slices', 'existing', pc_category, scan_idx + '~' + pc_filename))
            missing = load_ply(join(self.root_dir, 'slices', 'missing', pc_category, scan_idx + '~' + pc_filename))
            gt = load_ply(join(self.root_dir, pc_category, pc_filename))

        if self.is_random_rotated:
            existing = existing @ random_rotation_matrix
            missing = missing @ random_rotation_matrix
            gt = gt @ random_rotation_matrix

        return existing, missing, gt, synth_id_to_number[pc_category]

    @classmethod
    def _get_datasets_for_classes(cls, root_dir, split, classes=[], **kwargs):
        if not classes:
            if kwargs.get('use_pcn_model_list'):
                classes = ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243',
                           '04530566']
            else:
                classes = list(synth_id_to_category.keys())

        return {synth_id_to_category[category_id]: ShapeNetDataset(root_dir=root_dir,
                                                                   split=split,
                                                                   classes=[category_id], **kwargs)
                for category_id in classes}

    @classmethod
    def get_validation_datasets(cls, root_dir, classes=[], **kwargs):
        return cls._get_datasets_for_classes(root_dir, 'val', classes, **kwargs)

    @classmethod
    def get_test_datasets(cls, root_dir, classes=[], **kwargs):
        return cls._get_datasets_for_classes(root_dir, 'test', classes, **kwargs)
