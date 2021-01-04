from os import listdir
from os.path import join

import numpy as np
from datasets.base_dataset import BaseDataset
from utils.util import show_3d_cloud

# TODO replace
def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]


class RealDataNPYDataset(BaseDataset):

    def __init__(self, root_dir):
        super().__init__(root_dir)

        self.scenes = []
        self.objs = []

        for f in listdir(self.root_dir):
            if f.startswith('object'):
                self.objs.append(f)
            elif f.startswith('scen'):
                self.scenes.append(f)

    def _get_scales(self, pcd):
        axis_mins = np.min(pcd.T, axis=1)
        axis_maxs = np.max(pcd.T, axis=1)

        scale = np.max(axis_maxs - axis_mins)
        pcd_center = (axis_maxs + axis_mins) / 2

        return pcd_center, scale / 0.9

    def __getitem__(self, idx):
        pcd = np.load(join(self.root_dir, self.objs[idx])).astype(np.float32)
        pcd_center, scale = self._get_scales(pcd)
        pcd = (pcd - pcd_center) / scale
        return resample_pcd(pcd, 1024), 0, 0, idx

    def inverse_scale_to_scene(self, idx, scaled_pcd):
        scene = np.load(join(self.root_dir, self.scenes[idx])).astype(np.float32)
        pcd = np.load(join(self.root_dir, self.objs[idx])).astype(np.float32)
        pcd_center, scale = self._get_scales(pcd)
        scaled_pcd_center, scaled_pcd_scale = self._get_scales(scaled_pcd)
        return np.concatenate([scene, (scaled_pcd / scaled_pcd_scale * scale) + pcd_center])

    def inverse_scale(self, idx, scaled_pcd):
        pcd = np.load(join(self.root_dir, self.objs[idx])).astype(np.float32)
        pcd_center, scale = self._get_scales(pcd)
        scaled_pcd_center, scaled_pcd_scale = self._get_scales(scaled_pcd)
        return (scaled_pcd / scaled_pcd_scale * scale) + pcd_center

    def __len__(self):
        return len(self.objs)

    @classmethod
    def get_validation_datasets(cls, root_dir, classes=[], **kwargs):
        raise NotImplementedError
