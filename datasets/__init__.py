def get_datasets(config):
    dataset_name = config['name']

    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        train_dataset = ShapeNetDataset(root_dir=config['path'], classes=config['classes'], split='train',
                                        is_random_rotated=config['is_rotated'], use_pcn_model_list=True)
        val_dataset_dict = ShapeNetDataset.get_validation_datasets(root_dir=config['path'],
                                                                   classes=config['classes'],
                                                                   is_random_rotated=config['is_rotated'],
                                                                   use_pcn_model_list=True)
        test_dataset = ShapeNetDataset(root_dir=config['path'], classes=config['classes'], split='test',
                                       is_random_rotated=config['is_rotated'], use_pcn_model_list=True)
    elif dataset_name == 'completion':
        from datasets.shapenet_completion3d import ShapeNetCompletion3DDataset
        train_dataset = ShapeNetCompletion3DDataset(root_dir=config['path'], split='train', classes=config['classes'])
        val_dataset_dict = ShapeNetCompletion3DDataset.get_validation_datasets(config['data_dir'],
                                                                               classes=config['classes'])
        test_dataset = ShapeNetCompletion3DDataset(root_dir=config['data_dir'], split='test')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or `completion`. Got: `{dataset_name}`')

    return train_dataset, val_dataset_dict, test_dataset
