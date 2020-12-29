import logging
import re
from os import listdir, makedirs
from os.path import join, exists
from shutil import rmtree
from time import sleep
import random
import numpy as np

import torch


def setup_logging(log_dir):
    makedirs(log_dir, exist_ok=True)

    logpath = join(log_dir, 'log.txt')
    filemode = 'a' if exists(logpath) else 'w'

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=logpath,
                        filemode=filemode)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def prepare_results_dir(config, arch, experiment, dirs_to_create=('weights', 'samples', 'metrics')):
    output_dir = join(config['results_root'], arch, experiment, get_distribution_dir(config), config['dataset']['name'],
                      get_classes_dir(config), config['model_name'])
    if config['clean_results_dir']:
        if exists(output_dir):
            print('Attention! Cleaning results directory in 10 seconds!')
            sleep(10)
        rmtree(output_dir, ignore_errors=True)
    makedirs(output_dir, exist_ok=True)
    for dir_to_create in dirs_to_create:
        makedirs(join(output_dir, dir_to_create), exist_ok=True)
    return output_dir


def find_latest_epoch(dirpath):
    # Files with weights are in format ddddd_{D,E,G}.pth
    epoch_regex = re.compile(r'^(?P<n_epoch>\d+)_([DEG]|model)\.pth$')
    epochs_completed = []
    if exists(join(dirpath, 'weights')):
        dirpath = join(dirpath, 'weights')
    for f in listdir(dirpath):
        m = epoch_regex.match(f)
        if m:
            epochs_completed.append(int(m.group('n_epoch')))
    return max(epochs_completed) if epochs_completed else 0


def cuda_setup(cuda=False, gpu_idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(gpu_idx)
    return device


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # TODO Think about it
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_weights_dir(config):
    if config.get('weights_path'):
        weights_dir = config['weights_path']
    else:
        weights_dir = join(config['results_root'], config['arch'], 'training', get_distribution_dir(config),
                           config['dataset']['name'], get_classes_dir(config), 'weights')
        '''weights_dir = join(config['results_root'], config['arch'], 'training', get_distribution_dir(config),
                           config['dataset']['name'], get_classes_dir(config), config['model_name'], 'weights')'''
    if exists(weights_dir):
        return weights_dir
    raise FileNotFoundError(weights_dir)


def get_classes_dir(config):
    return 'all' if not config.get('classes') else '_'.join(config['classes'])


def get_distribution_dir(config):
    normed_str = ''
    if config['target_network_input']['normalization']['enable']:
        normalization_type = config['target_network_input']['normalization']['type']

        if normalization_type == 'progressive':
            norm_max_epoch = config['target_network_input']['normalization']['epoch']
            normed_str = 'normed_progressive_to_epoch_%d' % norm_max_epoch

    return '%s%s' % ('uniform', '_' + normed_str if normed_str else '')


def show_3d_cloud(points_cloud):
    import pptk
    # pptk.viewer(points_cloud).set(show_axis=False)
    pptk.viewer(points_cloud).set()


def replace_and_rename_pcd_file(source, dest):
    from shutil import copyfile
    model_ids = listdir(source)
    for model_id in model_ids:
        for sample in listdir(join(source, model_id)):
            for filename in listdir(join(source, model_id, sample)):
                copyfile(join(source, model_id, sample, filename), join(dest, f'{model_id}_{sample}_{filename}'))

