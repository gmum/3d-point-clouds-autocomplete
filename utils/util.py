import re
from os import listdir
from os.path import join, exists

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets.utils.shapenet_category_mapping import synth_id_to_category
from utils.pcutil import plot_3d_point_cloud


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


def get_classes_dir(config):
    return 'all' if not config.get('classes') else '_'.join(config['classes'])


def get_distribution_dir(config):
    normed_str = ''
    if config['target_network_input']['normalization']['enable']:
        if config['target_network_input']['normalization']['type'] == 'progressive':
            norm_max_epoch = config['target_network_input']['normalization']['epoch']
            normed_str = 'normed_progressive_to_epoch_%d' % norm_max_epoch

    return '%s%s' % ('uniform', '_' + normed_str if normed_str else '')


def get_model_name(config):
    model_name = ''
    encoders_num = 0
    real_size = config['full_model']['real_encoder']['output_size']
    random_size = config['full_model']['random_encoder']['output_size']

    if real_size > 0:
        encoders_num += 1
        model_name += str(real_size)

    if random_size > 0:
        encoders_num += 1
        model_name += 'x' + str(random_size) if real_size >0 else str(random_size)

    model_name = str(encoders_num) + 'e' + model_name

    model_name += config['training']['lr_scheduler']['type']

    for k, v in config['training']['lr_scheduler']['hyperparams'].items():
        model_name += '_' + k + str(v).replace(' ', '')

    return model_name


def show_3d_cloud(points_cloud):
    import pptk
    pptk.viewer(points_cloud).set()


def replace_and_rename_pcd_file(source, dest):
    from shutil import copyfile
    model_ids = listdir(source)
    for model_id in model_ids:
        for sample in listdir(join(source, model_id)):
            for filename in listdir(join(source, model_id, sample)):
                copyfile(join(source, model_id, sample, filename), join(dest, f'{model_id}_{sample}_{filename}'))


def get_filenames_by_cat(path) -> pd.DataFrame:
    filenames = []
    for category_id in synth_id_to_category.keys():
        for f in listdir(join(path, category_id)):
            if f not in ['.DS_Store']:
                filenames.append((category_id, f))
    return pd.DataFrame(filenames, columns=['category', 'filename'])


def save_plot(X, epoch, k, results_dir, t):
    fig = plot_3d_point_cloud(X[0], X[1], X[2], in_u_sphere=True, show=False, title=f'{t}_{k} epoch: {epoch}')
    fig_path = join(results_dir, f'{epoch}_{k}_{t}.png')
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]
