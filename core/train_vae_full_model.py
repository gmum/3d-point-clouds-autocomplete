import argparse
import json
import logging
from datetime import datetime
import shutil
from os.path import join, exists
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from datasets import get_datasets
from model.full_model import FullModel
from utils.pcutil import plot_3d_point_cloud
from utils.telegram_logging import TelegramLogger
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging, set_seed
from losses.champfer_loss import ChamferLoss


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


def save_plot(X, epoch, k, results_dir, t):
    fig = plot_3d_point_cloud(X[0], X[1], X[2], in_u_sphere=True, show=False, title=f'{t}_{k} epoch: {epoch}')
    fig_path = join(results_dir, 'samples', f'{epoch}_{k}_{t}.png')
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def main(config):
    set_seed(config['seed'])

    results_dir = prepare_results_dir(config, 'vae', 'training')
    starting_epoch = find_latest_epoch(results_dir) + 1

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    weights_path = join(results_dir, 'weights')
    metrics_path = join(results_dir, 'metrics')

    setup_logging(results_dir)
    log = logging.getLogger('vae')

    tg_log = None

    if config['use_telegram_logging']:
        tg_log = TelegramLogger(config['tg_bot_token'], config['tg_chat_id'])

    device = cuda_setup(config['cuda'], config['gpu'])

    log.info(f'Device variable: {device}')
    if device.type == 'cuda':
        log.info(f'Current CUDA device: {torch.cuda.current_device()}')

    train_dataset, val_dataset_dict, _ = get_datasets(config['dataset'])

    log.info("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['dataset']['classes'] else ','.join(config['classes']),
        len(train_dataset)))

    val_dataloaders_dict = {cat_name: DataLoader(cat_ds, batch_size=config['eval_batch_size'], shuffle=True,
                                                 num_workers=config['num_workers'], pin_memory=True)
                            for cat_name, cat_ds in val_dataset_dict.items()}

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
                                  num_workers=config['num_workers'], drop_last=True, pin_memory=True)

    full_model = FullModel(config['full_model']).to(device)
    full_model.apply(weights_init)

    reconstruction_loss = ChamferLoss().to(device)

    e_hn_optimizer = getattr(optim, config['optimizer']['E_HN']['type'])
    e_hn_optimizer = e_hn_optimizer(full_model.parameters(),
                                    **config['optimizer']['E_HN']['hyperparams'])

    scheduler = optim.lr_scheduler.StepLR(e_hn_optimizer, **config['scheduler'])

    log.info("Starting epoch: %s" % starting_epoch)
    if starting_epoch > 1:
        log.info("Loading weights...")

        full_model.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_model.pth')))

        e_hn_optimizer.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_O.pth')))

        scheduler.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_S.pth')))

        log.info("Loading losses...")
        losses_e = np.load(join(metrics_path, f'{starting_epoch - 1:05}_E.npy')).tolist()
        losses_kld = np.load(join(metrics_path, f'{starting_epoch - 1:05}_KLD.npy')).tolist()
        losses_eg = np.load(join(metrics_path, f'{starting_epoch - 1:05}_EG.npy')).tolist()
    else:
        log.info("First epoch")
        losses_e = []
        losses_kld = []
        losses_eg = []

    if config['target_network_input']['normalization']['enable']:
        normalization_type = config['target_network_input']['normalization']['type']
        assert normalization_type == 'progressive', 'Invalid normalization type'

    best_epoch_validation = -1
    best_validation_our_cd = 1e10

    target_network_input = None
    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()
        log.debug("Epoch: %s" % epoch)

        full_model.train()

        total_loss_all = 0.0
        total_loss_r = 0.0
        total_loss_kld = 0.0
        for i, point_data in enumerate(train_dataloader, 1):
            e_hn_optimizer.zero_grad()

            partial, remaining, gt, _ = point_data

            partial = partial.to(device)
            remaining = remaining.to(device)
            gt = gt.to(device)

            reconstruction, logvar, mu = full_model(partial, remaining, gt, epoch, device)

            loss_r = torch.mean(
                config['reconstruction_coef'] *
                reconstruction_loss(gt.permute(0, 2, 1) + 0.5, reconstruction.permute(0, 2, 1) + 0.5))

            loss_kld = 0.5 * (torch.exp(logvar) + torch.square(mu) - 1 - logvar).sum()
            loss_kld = torch.div(loss_kld, partial.shape[0])
            loss_all = loss_r + loss_kld

            total_loss_r += loss_r.item()
            total_loss_kld += loss_kld.item()
            total_loss_all += loss_all.item()

            loss_all.backward()
            e_hn_optimizer.step()

        scheduler.step()

        losses_e.append(total_loss_r)
        losses_kld.append(total_loss_kld)
        losses_eg.append(total_loss_all)

        log_string = f'[{epoch}/{config["max_epochs"]}] '\
                     f'Loss_ALL: {total_loss_all / i:.4f} '\
                     f'Loss_R: {total_loss_r / i:.4f} '\
                     f'Loss_E: {total_loss_kld / i:.4f} '\
                     f'Time: {datetime.now() - start_epoch_time}'

        log.info(log_string)

        partial = partial.cpu().numpy()
        gt = gt.cpu().numpy()
        reconstruction = reconstruction.detach().cpu().numpy()

        saved_plots = []
        for k in range(min(5, reconstruction.shape[0])):
            saved_plots.append(save_plot(partial[k], epoch, k, results_dir, 'cut'))
            saved_plots.append(save_plot(reconstruction[k], epoch, k, results_dir, 'reconstructed'))
            saved_plots.append(save_plot(gt[k], epoch, k, results_dir, 'real'))

        if config['use_telegram_logging']:
            tg_log.log_images(saved_plots[:9], log_string)

        if config['clean_weights_dir']:
            log.debug('Cleaning weights path: %s' % weights_path)
            shutil.rmtree(weights_path, ignore_errors=True)
            os.makedirs(weights_path, exist_ok=True)

        if epoch % config['val_frequency'] == 0:
            full_model.eval()
            val_losses = dict.fromkeys(val_dataset_dict.keys())
            with torch.no_grad():
                val_plots = []
                for cat_name, dl in val_dataloaders_dict.items():

                    total_loss_our_cd = 0.0

                    for i, point_data in enumerate(dl, 1):

                        partial, remaining, gt, _ = point_data
                        partial = partial.to(device)
                        remaining = remaining.to(device)
                        gt = gt.to(device)

                        reconstruction = full_model(partial, remaining, gt, epoch, device)

                        loss_our_cd = torch.mean(
                            config['reconstruction_coef'] *
                            reconstruction_loss(gt.permute(0, 2, 1), reconstruction.permute(0, 2, 1)))

                        total_loss_our_cd += loss_our_cd.item()

                    partial = partial.cpu().numpy()
                    gt = gt.cpu().numpy()
                    reconstruction = reconstruction.detach().cpu().numpy()

                    val_plots.append(save_plot(partial[0], epoch, cat_name, results_dir, 'val_partial'))
                    val_plots.append(save_plot(reconstruction[0], epoch, cat_name, results_dir, 'val_rec'))
                    val_plots.append(save_plot(gt[0], epoch, cat_name, results_dir, 'val_gt'))

                    val_losses[cat_name] = np.array([total_loss_our_cd/i])

                total = np.zeros(3)
                for v in val_losses.values():
                    total = np.add(total, v)
                val_losses['total'] = total / len(val_losses.keys())

                log_string = 'val results[0.05*our_cd, torch3d, 10^4*gr_loss]:\n'
                for k, v in val_losses.items():
                    log_string += k + ': ' + str(v) + '\n'

                if best_validation_our_cd > total[0]:
                    best_validation_our_cd = total[0]
                    best_validation_epoch = epoch
                    log_string += 'new best epoch ' + str(best_validation_epoch)

                log.info(log_string)
                if config['use_telegram_logging']:
                    chosen_plot_ids = np.random.choice(np.arange(8), 3, replace=False)  # TODO use num of classes
                    plots_to_log = []
                    for id in chosen_plot_ids:
                        plots_to_log.extend(val_plots[3*id:3*id+3])
                    tg_log.log_images(plots_to_log, log_string)

        if epoch % config['save_frequency'] == 0:
            log.debug('Saving data...')

            torch.save(full_model.state_dict(), join(weights_path, f'{epoch:05}_model.pth'))
            torch.save(e_hn_optimizer.state_dict(), join(weights_path, f'{epoch:05}_O.pth'))
            torch.save(scheduler.state_dict(), join(weights_path, f'{epoch:05}_S.pth'))

            np.save(join(metrics_path, f'{epoch:05}_E'), np.array(losses_e))
            np.save(join(metrics_path, f'{epoch:05}_KLD'), np.array(losses_kld))
            np.save(join(metrics_path, f'{epoch:05}_EG'), np.array(losses_eg))

            if config['use_telegram_logging']:
                tg_log.log("Epoch: %s saved" % epoch)


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
