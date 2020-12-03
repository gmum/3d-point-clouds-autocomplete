import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from datasets.shapenet import synth_id_to_category


class ShapeNetCompletionDataset(Dataset):

    def __init__(self, root_dir='/home/datasets/completion', split='train', classes=[], model_list=None):
        self.root_dir = root_dir

        self.split = split

        if split == 'train':
            self.list_path = os.path.join(root_dir, 'train.list')
        elif split == 'val':
            self.list_path = os.path.join(root_dir, 'val.list')
        else:
            self.list_path = os.path.join(root_dir, 'test.list')

        if model_list is None:
            with open(self.list_path) as file:
                if classes:
                    self.model_list = [line.strip() for line in file if line.strip().split('/')[0] in classes]
                else:
                    self.model_list = [line.strip() for line in file]
        else:
            self.model_list = model_list
        self.len = len(self.model_list)

    def __len__(self):
        return self.len

    def _load_h5(self, path):
        f = h5py.File(path, 'r')
        cloud_data = np.array(f['data'])
        f.close()
        return cloud_data.astype(np.float32)

    def __getitem__(self, index):
        model_name = self.model_list[index]
        partial = self._load_h5(os.path.join(self.root_dir, self.split, 'partial', model_name + '.h5'))
        if self.split != 'test':
            gt = self._load_h5(os.path.join(self.root_dir, self.split, 'gt', model_name + '.h5'))
        else:
            gt = partial
        return partial, gt, model_name

    @staticmethod
    def get_validation_datasets(root_dir, classes=[]):
        if not classes:
            classes = ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']

        list_path = os.path.join(root_dir, 'val.list')

        model_lists = dict.fromkeys(classes, [])

        with open(list_path) as file:
            for line in file:
                model_lists[line.strip().split('/')[0]].append(line.strip())
        return {synth_id_to_category[category_id]: ShapeNetCompletionDataset(root_dir=root_dir, split='val',
                                                                             model_list=model_list)
                for category_id, model_list in model_lists.items()}
