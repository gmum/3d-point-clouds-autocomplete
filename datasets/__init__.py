from datasets.shapenet import ShapeNetDataset
from datasets.shapenet_completion3d import ShapeNetCompletion3DDataset
from datasets.shapenet_3depn import ShapeNet3DEPNDataset


def get_datasets(config):
    dataset_name = config['name']
    if dataset_name == 'shapenet':
        train_dataset = ShapeNetDataset(root_dir=config['path'], classes=config['classes'], split='train',
                                        is_random_rotated=config['is_rotated'], num_samples=config['num_samples'],
                                        use_list_with_name='pcn')
        val_dataset_dict = ShapeNetDataset.get_validation_datasets(root_dir=config['path'],
                                                                   classes=config['classes'],
                                                                   is_random_rotated=config['is_rotated'],
                                                                   num_samples=config['num_samples'],
                                                                   use_list_with_name='pcn')
        test_dataset_dict = ShapeNetDataset.get_test_datasets(root_dir=config['path'],
                                                              classes=config['classes'],
                                                              is_random_rotated=config['is_rotated'],
                                                              num_samples=config['num_samples'],
                                                              use_list_with_name='pcn',
                                                              is_gen=config['gen_test_set'])
    elif dataset_name == 'completion':
        train_dataset = ShapeNetCompletion3DDataset(root_dir=config['path'], split='train', classes=config['classes'])
        val_dataset_dict = ShapeNetCompletion3DDataset.get_validation_datasets(config['path'], classes=config['classes'])
        test_dataset_dict = {'all': ShapeNetCompletion3DDataset(root_dir=config['path'], split='test')}
    elif dataset_name == '3depn':
        train_dataset = ShapeNet3DEPNDataset(root_dir=config['path'], split='train', classes=config['classes'])
        val_dataset_dict = ShapeNet3DEPNDataset.get_validation_datasets(config['path'], classes=config['classes'])
        test_dataset_dict = ShapeNet3DEPNDataset(root_dir=config['path'], split='test', classes=config['classes'])
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet`, `completion` or `3depn`. Got: `{dataset_name}`')

    return train_dataset, val_dataset_dict, test_dataset_dict
