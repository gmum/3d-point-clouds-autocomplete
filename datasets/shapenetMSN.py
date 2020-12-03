from os import makedirs, remove
from os.path import join, exists
from zipfile import ZipFile

import open3d as o3d
import requests
import torch
import numpy as np
import torch.utils.data as data
import os
import random


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]

class ShapeNet(data.Dataset):
    def __init__(self, root_dir, train=True, real_size=5000, npoints=8192, classes=[], num_of_samples=50):

        self.root_dir = root_dir

        self._maybe_download_data()

        if train:
            self.list_path = join(root_dir, 'train.list')
        else:
            self.list_path = join(root_dir, 'val.list')
        self.npoints = npoints

        self.num_of_samples = num_of_samples

        self.real_size = real_size
        self.train = train

        with open(self.list_path) as file:
            if classes:
                self.model_list = [line.strip().replace('/', '_') for line in file if line.strip().split('/')[0] in classes]
            else:
                self.model_list = [line.strip().replace('/', '_') for line in file]
        random.shuffle(self.model_list)
        self.len = len(self.model_list * self.num_of_samples)

    def _get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def _download_file_from_google_drive(self, session, dest, file_id):
        URL = "https://docs.google.com/uc?export=download"

        response = session.get(URL, params={'id': file_id}, stream=True)
        token = self._get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        with open(dest, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

    def _maybe_download_data(self):
        if exists(self.root_dir):
            return
        makedirs(self.root_dir, exist_ok=True)

        files_dict = {
            'val.list': '14KMGmlCNCk93LNoBpKu6mkTmSbGJMc6e',
            'val.zip': '1Bpfmz318Wzy6dMoxYCe1cGB5fFS0-pTN',
            'train.list': '1gdbmqP5cLedtQm9gP6fJ_0Bi8sL2FZQ4',
            'complete.zip': '1hlOZ0-WD_3Ape5jbDz09ZMKy2BGliuHa',
        }

        session = requests.Session()
        for filename, file_id in files_dict.items():
            path = join(self.root_dir, filename)
            self._download_file_from_google_drive(session, path, file_id)
            if filename.endswith('.zip'):
                with ZipFile(path, mode='r') as zip_f:
                    zip_f.extractall(self.root_dir)
                remove(path)

    def __getitem__(self, index):
        model_id = self.model_list[index // self.num_of_samples]
        scan_id = index % self.num_of_samples

        def read_pcd(filename):
            pcd = o3d.io.read_point_cloud(filename)
            return torch.from_numpy(np.array(pcd.points)).float()

        if self.train:
            partial = read_pcd(os.path.join(self.root_dir, 'train', model_id + '_%d.pcd' % scan_id))
            '''
            points = partial.numpy()
            points[:, 2] *= -1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(os.path.join(self.root_dir, 'train', model_id + '_%d.pcd' % scan_id), pcd)
            '''
        else:
            partial = read_pcd(os.path.join(self.root_dir, 'val', model_id + '_%d_denoised.pcd' % scan_id))
        complete = read_pcd(os.path.join(self.root_dir, 'complete', '%s.pcd' % model_id))
        return resample_pcd(partial, self.real_size), resample_pcd(complete, self.npoints), model_id

    def __len__(self):
        return self.len


def get_category_val_datasets(classes=[], **kwargs):
    categories = ['vessel', 'cabinet', 'table', 'airplane', 'car', 'chair', 'sofa', 'lamp']
    from datasets.shapenet import category_to_synth_id
    return {cat_name: ShapeNet(train=False, num_of_samples=50, classes=[category_to_synth_id[cat_name]], **kwargs)
            for cat_name in categories}
