from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, root_dir, split='train', classes=[]):
        self.root_dir = root_dir
        self.split = split

    @classmethod
    def get_validation_datasets(cls, root_dir, classes=[], **kwargs):
        raise NotImplementedError
