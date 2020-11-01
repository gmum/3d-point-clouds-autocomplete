import argparse
import json
import re
from datetime import datetime
from os import listdir
from os.path import join
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models import aae

from utils.points import generate_points
from utils.metrics import jsd_between_point_cloud_sets
from utils.util import set_seed, cuda_setup, get_weights_dir, find_latest_epoch


def _get_epochs_by_regex(path, regex):
    reg = re.compile(regex)
    return {int(w[:5]) for w in listdir(path) if reg.match(w)}


def jsd(config, weights_path, device):
    print('Evaluating Jensen-Shannon divergences on validation set on all saved epochs.')

    # Find all epochs that have saved model weights
    e_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{5})_E\.pth')
    g_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{5})_G\.pth')
    epochs = sorted(e_epochs.intersection(g_epochs))

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], split='valid')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet`. Got: `{dataset_name}`')

    classes_selected = ('all' if not config['classes']
                        else ','.join(config['classes']))
    print(f'Valid dataset. Selected {classes_selected} classes. Loaded {len(dataset)} '
          f'samples.')

    distribution = config['metrics']['distribution']
    assert distribution in ['normal', 'beta'], 'Invalid distribution. Choose normal or beta'

    #
    # Models
    #
    hyper_network = aae.HyperNetwork(config, device).to(device)

    hyper_network.eval()

    num_samples = len(dataset.point_clouds_names_valid)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    X, _ = next(iter(data_loader))
    X = X.to(device)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    noise = torch.zeros(3 * X.shape[0], config['z_size']).to(device)

    results = {}

    n_last_epochs = config['metrics'].get('jsd_how_many_last_epochs', -1)
    epochs = epochs[-n_last_epochs:] if n_last_epochs > 0 else epochs
    print(f'Testing epochs: {epochs}')

    for epoch in reversed(epochs):
        try:
            hyper_network.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

            start_clock = datetime.now()

            # We average JSD computation from 3 independent trials.
            js_results = []
            for _ in range(3):
                if distribution == 'normal':
                    noise.normal_(config['metrics']['normal_mu'], config['metrics']['normal_std'])
                elif distribution == 'beta':
                    noise_np = np.random.beta(config['metrics']['beta_a'], config['metrics']['beta_b'], noise.shape)
                    noise = torch.tensor(noise_np).float().round().to(device)

                with torch.no_grad():
                    target_networks_weights = hyper_network(noise)

                    X_rec = torch.zeros(3 * X.shape[0], X.shape[1], X.shape[2]).to(device)

                    # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
                    if X_rec.size(-1) == 3:
                        X_rec.transpose_(X_rec.dim() - 2, X_rec.dim() - 1)

                    for j, target_network_weights in enumerate(target_networks_weights):
                        target_network = aae.TargetNetwork(config, target_network_weights).to(device)

                        target_network_input = generate_points(config=config, epoch=epoch,
                                                               size=(X_rec.shape[2], X_rec.shape[1]))

                        X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

                jsd = jsd_between_point_cloud_sets(X.cpu().numpy(), torch.transpose(X_rec, 1, 2).cpu().numpy())
                js_results.append(jsd)

            js_result = np.mean(js_results)
            print(f'Epoch: {epoch} JSD: {js_result: .6f} '
                  f'Time: {datetime.now() - start_clock}')
            results[epoch] = js_result
        except KeyboardInterrupt:
            print(f'Interrupted during epoch: {epoch}')
            break

    results = pd.DataFrame.from_dict(results, orient='index', columns=['jsd'])
    print(f"Minimum JSD at epoch {results.idxmin()['jsd']}: "
          f"{results.min()['jsd']: .6f}")

    return results.idxmin()['jsd'], results.min()['jsd']


def minimum_matching_distance(config, weights_path, device):
    from utils.metrics import EMD_CD
    print('Minimum Matching Distance (MMD) Test split')
    epoch = find_latest_epoch(weights_path)
    print(f'Last Epoch: {epoch}')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], split='test')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not config['classes']
                        else ','.join(config['classes']))
    print(f'Test dataset. Selected {classes_selected} classes. Loaded {len(dataset)} '
          f'samples.')

    #
    # Models
    #
    hyper_network = aae.HyperNetwork(config, device).to(device)
    encoder = aae.Encoder(config).to(device)

    hyper_network.eval()
    encoder.eval()

    num_samples = len(dataset.point_clouds_names_test)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    encoder.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))
    hyper_network.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

    result = {}

    for point_data in data_loader:

        X, _ = point_data
        X = X.to(device)

        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X.size(-1) == 3:
            X.transpose_(X.dim() - 2, X.dim() - 1)

        with torch.no_grad():
            z_a, _, _ = encoder(X)
            target_networks_weights = hyper_network(z_a)

            X_rec = torch.zeros(X.shape).to(device)
            for j, target_network_weights in enumerate(target_networks_weights):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)

                target_network_input = generate_points(config=config, epoch=epoch, size=(X.shape[2], X.shape[1]))

                X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

            for k, v in EMD_CD(torch.transpose(X, 1, 2).contiguous(),
                               torch.transpose(X_rec, 1, 2).contiguous(), X.shape[0]).items():
                result[k] = result.get(k, 0.0) + v.item()

    print(result)


def all_metrics(config, weights_path, device, epoch, jsd_value):
    from utils.metrics import compute_all_metrics
    print('All metrics')
    if epoch is None:
        print('Finding latest epoch...')
        epoch = find_latest_epoch(weights_path)
        print(f'Epoch: {epoch}')

    if jsd_value is not None:
        print(f'Best Epoch selected via mimnimal JSD: {epoch}')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], split='test')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not config['classes']
                        else ','.join(config['classes']))
    print(f'Test dataset. Selected {classes_selected} classes. Loaded {len(dataset)} '
          f'samples.')

    distribution = config['metrics']['distribution']
    assert distribution in ['normal', 'beta'], 'Invalid distribution. Choose normal or beta'

    #
    # Models
    #
    hyper_network = aae.HyperNetwork(config, device).to(device)

    hyper_network.eval()

    data_loader = DataLoader(dataset, batch_size=32,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    hyper_network.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

    result = {}
    size = 0
    start_clock = datetime.now()
    for point_data in data_loader:

        X, _ = point_data
        X = X.to(device)

        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X.size(-1) == 3:
            X.transpose_(X.dim() - 2, X.dim() - 1)

        with torch.no_grad():
            noise = torch.zeros(X.shape[0], config['z_size']).to(device)
            if distribution == 'normal':
                noise.normal_(config['metrics']['normal_mu'], config['metrics']['normal_std'])
            elif distribution == 'beta':
                noise_np = np.random.beta(config['metrics']['beta_a'], config['metrics']['beta_b'], noise.shape)
                noise = torch.tensor(noise_np).float().round().to(device)

            target_networks_weights = hyper_network(noise)

            X_rec = torch.zeros(X.shape).to(device)
            for j, target_network_weights in enumerate(target_networks_weights):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)

                target_network_input = generate_points(config=config, epoch=epoch, size=(X.shape[2], X.shape[1]))

                X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

            for k, v in compute_all_metrics(torch.transpose(X, 1, 2).contiguous(),
                                            torch.transpose(X_rec, 1, 2).contiguous(), X.shape[0]).items():
                result[k] = (size * result.get(k, 0.0) + X.shape[0] * v.item()) / (size + X.shape[0])

        size += X.shape[0]

    result['jsd'] = jsd_value
    print(f'Time: {datetime.now() - start_clock}')
    print(f'Result:')
    for k, v in result.items():
        print(f'{k}: {v}')


def main(config):
    set_seed(config['seed'])
    weights_path = get_weights_dir(config)

    device = cuda_setup(config['cuda'], config['gpu'])
    print(f'Device variable: {device}')
    if device.type == 'cuda':
        print(f'Current CUDA device: {torch.cuda.current_device()}')

    print('\n')
    all_metrics(config, weights_path, device, None, None)
    print('\n')

    set_seed(config['seed'])
    jsd_epoch, jsd_value = jsd(config, weights_path, device)
    print('\n')

    set_seed(config['seed'])
    all_metrics(config, weights_path, device, jsd_epoch, jsd_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, help='File path for evaluation config')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    main(evaluation_config)
