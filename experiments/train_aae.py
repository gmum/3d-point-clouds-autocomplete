import argparse
import json
import logging
from datetime import datetime
import shutil
from itertools import chain
from os.path import join, exists
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import grad
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from models import aae
from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging, set_seed
from utils.points import generate_points

cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def feature_transform_regularization(t: torch.Tensor) -> torch.Tensor:
    """Regularization used in PointNet for feature transformation T-Net weights to obtain orthogonal matrix.

    Args:
        t (torch.Tensor): Feature transformation matrix

    Returns:
        (float): Regularization loss
    """
    N, D, _ = t.size()
    I = torch.eye(D).unsqueeze(0).to(t.device).type_as(t)
    loss = torch.norm(torch.bmm(t, t.transpose(2, 1)) - I, dim=(1, 2), p=2)
    return loss


def main(config):
    set_seed(config['seed'])

    results_dir = prepare_results_dir(config, 'aae', 'training')
    starting_epoch = find_latest_epoch(results_dir) + 1

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir)
    log = logging.getLogger('aae')

    device = cuda_setup(config['cuda'], config['gpu'])
    log.info(f'Device variable: {device}')
    if device.type == 'cuda':
        log.info(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')
    metrics_path = join(results_dir, 'metrics')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'])
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.info("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']),
        len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                   shuffle=config['shuffle'],
                                   num_workers=config['num_workers'],
                                   pin_memory=True)

    pointnet = config.get('pointnet', False)
    #
    # Models
    #
    hyper_network = aae.HyperNetwork(config, device).to(device)

    if pointnet:
        from models.pointnet import PointNet
        encoder = PointNet(config).to(device)
        # PointNet initializes it's own weights during instance creation
    else:
        encoder = aae.EncoderForRandomPoints(config).to(device)
        encoder.apply(weights_init)

    discriminator = aae.Discriminator(config).to(device)

    hyper_network.apply(weights_init)
    discriminator.apply(weights_init)

    if config['reconstruction_loss'].lower() == 'chamfer':
        if pointnet:
            from utils.metrics import chamfer_distance
            reconstruction_loss = chamfer_distance
        else:
            from losses.champfer_loss import ChamferLoss
            reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        from utils.metrics import earth_mover_distance
        reconstruction_loss = earth_mover_distance
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    #
    # Optimizers
    #
    e_hn_optimizer = getattr(optim, config['optimizer']['E_HN']['type'])
    e_hn_optimizer = e_hn_optimizer(chain(encoder.parameters(), hyper_network.parameters()),
                                    **config['optimizer']['E_HN']['hyperparams'])

    discriminator_optimizer = getattr(optim, config['optimizer']['D']['type'])
    discriminator_optimizer = discriminator_optimizer(discriminator.parameters(),
                                                      **config['optimizer']['D']['hyperparams'])

    log.info("Starting epoch: %s" % starting_epoch)
    if starting_epoch > 1:
        log.info("Loading weights...")
        hyper_network.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_G.pth')))
        encoder.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_E.pth')))
        discriminator.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_D.pth')))

        e_hn_optimizer.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_EGo.pth')))

        discriminator_optimizer.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_Do.pth')))

        log.info("Loading losses...")
        losses_e = np.load(join(metrics_path, f'{starting_epoch - 1:05}_E.npy')).tolist()
        losses_g = np.load(join(metrics_path, f'{starting_epoch - 1:05}_G.npy')).tolist()
        losses_eg = np.load(join(metrics_path, f'{starting_epoch - 1:05}_EG.npy')).tolist()
        losses_d = np.load(join(metrics_path, f'{starting_epoch - 1:05}_D.npy')).tolist()
    else:
        log.info("First epoch")
        losses_e = []
        losses_g = []
        losses_eg = []
        losses_d = []

    normalize_points = config['target_network_input']['normalization']['enable']
    if normalize_points:
        normalization_type = config['target_network_input']['normalization']['type']
        assert normalization_type == 'progressive', 'Invalid normalization type'

    target_network_input = None
    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()
        log.debug("Epoch: %s" % epoch)
        hyper_network.train()
        encoder.train()
        discriminator.train()

        total_loss_all = 0.0
        total_loss_reconstruction = 0.0
        total_loss_encoder = 0.0
        total_loss_discriminator = 0.0
        total_loss_regularization = 0.0
        for i, point_data in enumerate(points_dataloader, 1):

            X, _ = point_data
            X = X.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            if pointnet:
                _, feature_transform, codes = encoder(X)
            else:
                codes, _, _ = encoder(X)

            # discriminator training
            noise = torch.empty(codes.shape[0], config['remaining_size']).normal_(mean=config['normal_mu'],
                                                                          std=config['normal_std']).to(device)
            synth_logit = discriminator(codes)
            real_logit = discriminator(noise)
            if config.get('wasserstein', True):
                loss_discriminator = torch.mean(synth_logit) - torch.mean(real_logit)

                alpha = torch.rand(codes.shape[0], 1).to(device)
                differences = codes - noise
                interpolates = noise + alpha * differences
                disc_interpolates = discriminator(interpolates)

                # gradient_penalty_function
                gradients = grad(
                    outputs=disc_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(disc_interpolates).to(device),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
                slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
                gradient_penalty = ((slopes - 1) ** 2).mean()
                loss_gp = config['gradient_penalty_coef'] * gradient_penalty
                loss_discriminator += loss_gp
            else:
                # An alternative is a = -1, b = 1 iff c = 0
                a = 0.0
                b = 1.0
                loss_discriminator = 0.5 * ((real_logit - b)**2 + (synth_logit - a)**2)

            discriminator_optimizer.zero_grad()
            discriminator.zero_grad()

            loss_discriminator.backward(retain_graph=True)
            total_loss_discriminator += loss_discriminator.item()
            discriminator_optimizer.step()

            # hyper network training
            target_networks_weights = hyper_network(codes)

            X_rec = torch.zeros(X.shape).to(device)
            for j, target_network_weights in enumerate(target_networks_weights):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)

                if not config['target_network_input']['constant'] or target_network_input is None:
                    target_network_input = generate_points(config=config, epoch=epoch, size=(X.shape[2], X.shape[1]))

                X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

            if pointnet:
                loss_reconstruction = config['reconstruction_coef'] * \
                                      reconstruction_loss(torch.transpose(X, 1, 2).contiguous(),
                                                          torch.transpose(X_rec, 1, 2).contiguous(),
                                                          batch_size=X.shape[0]).mean()
            else:
                loss_reconstruction = torch.mean(
                    config['reconstruction_coef'] *
                    reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                        X_rec.permute(0, 2, 1) + 0.5))

            # encoder training
            synth_logit = discriminator(codes)
            if config.get('wasserstein', True):
                loss_encoder = -torch.mean(synth_logit)
            else:
                # An alternative is c = 0 iff a = -1, b = 1
                c = 1.0
                loss_encoder = 0.5 * (synth_logit - c)**2

            if pointnet:
                regularization_loss = config['feature_regularization_coef'] * \
                                      feature_transform_regularization(feature_transform).mean()
                loss_all = loss_reconstruction + loss_encoder + regularization_loss
            else:
                loss_all = loss_reconstruction + loss_encoder

            e_hn_optimizer.zero_grad()
            encoder.zero_grad()
            hyper_network.zero_grad()

            loss_all.backward()
            e_hn_optimizer.step()

            total_loss_reconstruction += loss_reconstruction.item()
            total_loss_encoder += loss_encoder.item()
            total_loss_all += loss_all.item()

            if pointnet:
                total_loss_regularization += regularization_loss.item()

        log.info(
            f'[{epoch}/{config["max_epochs"]}] '
            f'Total_Loss: {total_loss_all / i:.4f} '
            f'Loss_R: {total_loss_reconstruction / i:.4f} '
            f'Loss_E: {total_loss_encoder / i:.4f} '
            f'Loss_D: {total_loss_discriminator / i:.4f} '
            f'Time: {datetime.now() - start_epoch_time}'
        )

        if pointnet:
            log.info(f'Loss_Regularization: {total_loss_regularization / i:.4f}')

        losses_e.append(total_loss_reconstruction)
        losses_g.append(total_loss_encoder)
        losses_eg.append(total_loss_all)
        losses_d.append(total_loss_discriminator)

        #
        # Save intermediate results
        #
        X = X.cpu().numpy()
        X_rec = X_rec.detach().cpu().numpy()

        for k in range(min(5, X_rec.shape[0])):
            fig = plot_3d_point_cloud(X_rec[k][0], X_rec[k][1], X_rec[k][2], in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(join(results_dir, 'samples', f'{epoch}_{k}_reconstructed.png'))
            plt.close(fig)

            fig = plot_3d_point_cloud(X[k][0], X[k][1], X[k][2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'samples', f'{epoch}_{k}_real.png'))
            plt.close(fig)

        if config['clean_weights_dir']:
            log.debug('Cleaning weights path: %s' % weights_path)
            shutil.rmtree(weights_path, ignore_errors=True)
            os.makedirs(weights_path, exist_ok=True)

        if epoch % config['save_frequency'] == 0:
            log.debug('Saving data...')

            torch.save(hyper_network.state_dict(), join(weights_path, f'{epoch:05}_G.pth'))
            torch.save(encoder.state_dict(), join(weights_path, f'{epoch:05}_E.pth'))
            torch.save(e_hn_optimizer.state_dict(), join(weights_path, f'{epoch:05}_EGo.pth'))
            torch.save(discriminator.state_dict(), join(weights_path, f'{epoch:05}_D.pth'))
            torch.save(discriminator_optimizer.state_dict(), join(weights_path, f'{epoch:05}_Do.pth'))

            np.save(join(metrics_path, f'{epoch:05}_E'), np.array(losses_e))
            np.save(join(metrics_path, f'{epoch:05}_G'), np.array(losses_g))
            np.save(join(metrics_path, f'{epoch:05}_EG'), np.array(losses_eg))
            np.save(join(metrics_path, f'{epoch:05}_D'), np.array(losses_d))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
