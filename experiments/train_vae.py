import argparse
import json
import logging
from datetime import datetime
import shutil
from itertools import chain
from os.path import join, exists, basename
from zipfile import ZipFile

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import aae
from utils.pcutil import plot_3d_point_cloud
from utils.telegram_logging import TelegramLogger
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging, set_seed
from utils.points import generate_points
from chamferdist import ChamferDistance
from losses.champfer_loss import ChamferLoss
from losses.chamfer_dist import ChamferDistance as CD


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

    setup_logging(results_dir)
    log = logging.getLogger('vae')

    tg_log = None

    if config['use_telegram_logging']:
        tg_log = TelegramLogger(config['tg_bot_token'], config['tg_chat_id'])

    device = cuda_setup(config['cuda'], config['gpu'])

    log.info(f'Device variable: {device}')
    if device.type == 'cuda':
        log.info(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')
    metrics_path = join(results_dir, 'metrics')

    #
    # Dataset
    #
    dataset_name = config['dataset_name'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'],
                                  is_sliced=True, is_random_rotated=True)
        val_dataset_dict = ShapeNetDataset.get_validation_datasets(root_dir=config['data_dir'],
                                                                   classes=config['classes'],
                                                                   is_sliced=True, is_random_rotated=True)
    elif dataset_name == 'shapenet_msn':
        from datasets.shapenet_msn import ShapeNet
        dataset = ShapeNet(root_dir=config['data_dir'], split='train',
                           real_size=config['real_size'],
                           npoints=config['n_points'],
                           num_of_samples=config['num_of_samples'],
                           classes=config['classes'])
        # TODO add validation datasets
    elif dataset_name == 'completion':
        from datasets.shapenet_completion3d import ShapeNetCompletion3DDataset
        dataset = ShapeNetCompletion3DDataset(root_dir=config['data_dir'], split='train', classes=config['classes'])
        val_dataset_dict = ShapeNetCompletion3DDataset.get_validation_datasets(config['data_dir'], classes=config['classes'])
        test_dataset = ShapeNetCompletion3DDataset(root_dir=config['data_dir'], split='test')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.info("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']),
        len(dataset)))

    val_dataloaders_dict = {cat_name: DataLoader(cat_ds, batch_size=config['eval_batch_size'], shuffle=True,
                                                 num_workers=config['num_workers'], pin_memory=True)
                            for cat_name, cat_ds in val_dataset_dict.items()}

    train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
                                  num_workers=config['num_workers'], drop_last=True, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=config['test']['batch_size'], shuffle=config['shuffle'],
                                 num_workers=config['num_workers'])

    hyper_network = aae.HyperNetwork(config, device).to(device)
    real_data_encoder = aae.EncoderForRealPoints(config).to(device)

    hyper_network.apply(weights_init)
    real_data_encoder.apply(weights_init)

    torch3d_cd_loss = ChamferDistance().to(device)
    gr_cd_loss = CD().to(device)

    # TODO refactor
    if config['reconstruction_loss'].lower() == 'chamfer':
        loss_id = 0
        reconstruction_loss = ChamferLoss().to(device)
    # elif config['reconstruction_loss'].lower() == 'emd':
    #     loss_id = 1
    #     reconstruction_loss = losses_functions['msn emd']
    elif config['reconstruction_loss'].lower() == 'chamferdist':
        pass
        # loss_id = 2
        # reconstruction_loss = losses_functions['chamfer dist']
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    #
    # Optimizers #Fixme Change here
    #
    e_hn_optimizer = getattr(optim, config['optimizer']['E_HN']['type'])
    e_hn_optimizer = e_hn_optimizer(chain(# encoder.parameters(),
                                          real_data_encoder.parameters(),
                                          hyper_network.parameters()),
                                    **config['optimizer']['E_HN']['hyperparams'])

    if config['test']['execute']:
        test_epoch = config['test']['epoch']
        hyper_network.load_state_dict(torch.load(
            join(weights_path, f'{test_epoch:05}_G.pth')))
        real_data_encoder.load_state_dict(torch.load(
            join(weights_path, f'{test_epoch:05}_ER.pth')))

        hyper_network.eval()
        real_data_encoder.eval()

        benchmark_submission_dir = join(config['results_root'], 'benchmark', 'shapenet', 'test', 'partial', 'all')
        os.makedirs(benchmark_submission_dir, exist_ok=True)

        submission_zip = ZipFile('submission.zip', 'w')

        for i, point_data in tqdm(enumerate(test_dataloader, 1), total=len(test_dataloader)):
            partial, _, model_id = point_data
            partial = partial.to(device)

            if partial.size(-1) == 3:
                partial.transpose_(partial.dim() - 2, partial.dim() - 1)

            real_mu = real_data_encoder(partial)

            target_networks_weights = hyper_network(real_mu)

            X_rec = torch.zeros(partial.shape).to(device)
            for j, target_network_weights in enumerate(target_networks_weights):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)
                target_network_input = generate_points(config=config, epoch=test_epoch, size=(partial.shape[2],
                                                                                         partial.shape[1]))
                X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

            X_rec = X_rec.detach().cpu()
            for idx, x in enumerate(X_rec.permute(0, 2, 1)):

                ofile = join(benchmark_submission_dir, model_id[idx].split('/')[-1] + '.h5')
                with h5py.File(ofile, "w") as f:
                    f.create_dataset("data", data=x.numpy())
                    f.close()
                submission_zip.write(ofile, 'all/'+basename(ofile))

        if config['test']['only']:
            exit(0)

    log.info("Starting epoch: %s" % starting_epoch)
    if starting_epoch > 1:
        log.info("Loading weights...")
        hyper_network.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_G.pth')))
        # encoder.load_state_dict(torch.load(
        #     join(weights_path, f'{starting_epoch - 1:05}_E.pth')))
        real_data_encoder.load_state_dict(torch.load(
             join(weights_path, f'{starting_epoch - 1:05}_ER.pth')))
        e_hn_optimizer.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_EGo.pth')))

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

    scheduler = optim.lr_scheduler.StepLR(e_hn_optimizer, **config['scheduler'])

    target_network_input = None
    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()
        log.debug("Epoch: %s" % epoch)

        hyper_network.train()
        # encoder.train()
        real_data_encoder.train()

        total_loss_all = 0.0
        total_loss_r = 0.0
        total_loss_kld = 0.0

        for i, point_data in tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader)):
            e_hn_optimizer.zero_grad()

            partial, gt, _ = point_data
            partial = partial.to(device)
            gt = gt.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if partial.size(-1) == 3:
                partial.transpose_(partial.dim() - 2, partial.dim() - 1)

            if gt.size(-1) == 3:
                gt.transpose_(gt.dim() - 2, gt.dim() - 1)

            # codes, mu, logvar = encoder(partial)
            real_mu = real_data_encoder(partial)

            # target_networks_weights = hyper_network(torch.cat([codes, real_mu], 1))

            target_networks_weights = hyper_network(real_mu)

            X_rec = torch.zeros(gt.shape).to(device)
            for j, target_network_weights in enumerate(target_networks_weights):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)

                if not config['target_network_input']['constant'] or target_network_input is None:
                    target_network_input = generate_points(config=config, epoch=epoch, size=(gt.shape[2], gt.shape[1]))

                X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

            loss_r = torch.mean(
                config['reconstruction_coef'] *
                reconstruction_loss(gt.permute(0, 2, 1) + 0.5, X_rec.permute(0, 2, 1) + 0.5))
            # TODO refactor
            '''
            if loss_id == 0:
                
            elif loss_id == 1:
                dist, _ = reconstruction_loss(gt.permute(0, 2, 1) + 0.5, X_rec.permute(0, 2, 1) + 0.5, 0.005, 50)
                loss_r = torch.mean(config['reconstruction_coef'] * torch.sqrt(dist))
            elif loss_id == 2:
                loss_r = reconstruction_loss(X_rec.permute(0, 2, 1) + 0.5, gt.permute(0, 2, 1) + 0.5, reduction='mean')
            '''

            # loss_kld = 0.5 * (torch.exp(logvar) + torch.square(mu) - 1 - logvar).sum()
            # loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # loss_kld = torch.div(loss_kld, partial.shape[0])
            loss_all = loss_r # + loss_kld

            total_loss_r += loss_r.item()
            # total_loss_kld += loss_kld.item()
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
        X_rec = X_rec.detach().cpu().numpy()

        saved_plots = []
        for k in range(min(5, X_rec.shape[0])):
            saved_plots.append(save_plot(partial[k], epoch, k, results_dir, 'cut'))
            saved_plots.append(save_plot(X_rec[k], epoch, k, results_dir, 'reconstructed'))
            saved_plots.append(save_plot(gt[k], epoch, k, results_dir, 'real'))

        if config['use_telegram_logging']:
            tg_log.log_images(saved_plots[:9], log_string)

        if config['clean_weights_dir']:
            log.debug('Cleaning weights path: %s' % weights_path)
            shutil.rmtree(weights_path, ignore_errors=True)
            os.makedirs(weights_path, exist_ok=True)

        if epoch % config['val_frequency'] == 0:
            hyper_network.eval()
            # encoder.eval()
            real_data_encoder.eval()
            val_losses = dict.fromkeys(val_dataset_dict.keys())
            with torch.no_grad():
                val_plots = []
                for cat_name, dl in val_dataloaders_dict.items():

                    total_loss_our_cd = 0.0
                    total_loss_torch3d_cd = 0.0
                    total_loss_gr_cd = 0.0

                    for i, point_data in enumerate(dl, 1):

                        partial, gt, _ = point_data
                        partial = partial.to(device)
                        gt = gt.to(device)

                        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
                        if partial.size(-1) == 3:
                            partial.transpose_(partial.dim() - 2, partial.dim() - 1)

                        if gt.size(-1) == 3:
                            gt.transpose_(gt.dim() - 2, gt.dim() - 1)

                        # _, random_mu, _ = encoder(partial)

                        # fixed_noise = torch.zeros(partial.shape[0], config['random_encoder_output_size'])  # .normal_(mean=0.0, std=0.015)  # TODO consider about mean and std
                        real_mu = real_data_encoder(partial)

                        # target_networks_weights = hyper_network(torch.cat([random_mu, real_mu], 1))

                        target_networks_weights = hyper_network(real_mu)

                        X_rec = torch.zeros(gt.shape).to(device)
                        for j, target_network_weights in enumerate(target_networks_weights):
                            target_network = aae.TargetNetwork(config, target_network_weights).to(device)
                            target_network_input = generate_points(config=config, epoch=epoch, size=(gt.shape[2],
                                                                                                     gt.shape[1]))
                            X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

                        loss_our_cd = torch.mean(
                            config['reconstruction_coef'] *
                            reconstruction_loss(gt.permute(0, 2, 1), X_rec.permute(0, 2, 1)))

                        loss_torch3d_cd = torch3d_cd_loss(gt.permute(0, 2, 1), X_rec.permute(0, 2, 1),
                                                          bidirectional=True)
                        loss_gr_cd = gr_cd_loss(gt.permute(0, 2, 1), X_rec.permute(0, 2, 1))

                        total_loss_our_cd += loss_our_cd.item()
                        total_loss_torch3d_cd += loss_torch3d_cd.item()
                        total_loss_gr_cd += loss_gr_cd.item()

                    partial = partial.cpu().numpy()
                    gt = gt.cpu().numpy()
                    X_rec = X_rec.detach().cpu().numpy()

                    val_plots.append(save_plot(partial[0], epoch, cat_name, results_dir, 'val_partial'))
                    val_plots.append(save_plot(X_rec[0], epoch, cat_name, results_dir, 'val_rec'))
                    val_plots.append(save_plot(gt[0], epoch, cat_name, results_dir, 'val_gt'))

                    val_losses[cat_name] = np.array([total_loss_our_cd/i, total_loss_torch3d_cd/i, 10000*total_loss_gr_cd/i])

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

        if (epoch % config['save_frequency'] == 0) or (epoch == best_validation_epoch):
            log.debug('Saving data...')

            torch.save(hyper_network.state_dict(), join(weights_path, f'{epoch:05}_G.pth'))
            # torch.save(encoder.state_dict(), join(weights_path, f'{epoch:05}_E.pth'))
            torch.save(real_data_encoder.state_dict(), join(weights_path, f'{epoch:05}_ER.pth'))
            torch.save(e_hn_optimizer.state_dict(), join(weights_path, f'{epoch:05}_EGo.pth'))

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
