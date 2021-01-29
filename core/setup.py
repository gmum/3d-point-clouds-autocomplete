import logging
from os import makedirs
from os.path import join, exists

import random
import numpy as np

import torch

from utils.util import get_classes_dir, get_distribution_dir, get_model_name


def seed_setup(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_results_dir_path(config, mode):
    return join(config['results_root'], mode, get_distribution_dir(config['full_model']),
                config['dataset']['name'], get_classes_dir(config['dataset']), get_model_name(config))


def results_dir_setup(dir_path, dirs_to_create=('weights', 'samples', 'metrics')):
    makedirs(dir_path, exist_ok=True)
    for dir_to_create in dirs_to_create:
        makedirs(join(dir_path, dir_to_create), exist_ok=True)
    return dir_path


def logging_setup(log_dir):
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


def cuda_setup(cuda=False, gpu_idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(gpu_idx)
    return device


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def restore_model_state(weights_path, metrics_path, gpu_id, epoch, restore_policy, full_model, optimizer=None,
                        scheduler=None):

    if restore_policy == "latest":
        pass
    elif restore_policy == "best_val":
        val_losses = np.load(join(metrics_path, f'{epoch:05}_val.npy'), allow_pickle=True)
        epoch = np.argmin(val_losses) + 1
    else:
        # TODO handle value error
        epoch = int(restore_policy)
    # full_model.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_model.pth')))

    full_model.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_model.pth'),
                                          map_location='cuda:' + str(gpu_id)))

    if optimizer is not None:
        optimizer.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_O.pth')))

    if scheduler is not None:
        scheduler.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_S.pth')))
    return epoch


def restore_metrics(metrics_path, epoch):
    train_losses = np.load(join(metrics_path, f'{epoch:05}_train.npy'), allow_pickle=True)
    val_losses = np.load(join(metrics_path, f'{epoch:05}_val.npy'), allow_pickle=True)
    return train_losses.tolist(), val_losses.tolist(), np.min(val_losses)
