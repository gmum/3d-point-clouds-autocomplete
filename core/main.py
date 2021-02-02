import json
import logging

from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.arg_parser import parse_config
from core.epoch_loops import train_epoch, val_epoch
from core.experiments import experiment_functions_dict
from datasets import get_datasets
from model.full_model import FullModel
from losses.champfer_loss import ChamferLoss

from core.setup import seed_setup, logging_setup, cuda_setup, results_dir_setup, restore_model_state, \
    get_results_dir_path, restore_metrics, weights_init
from utils.telegram_logging import TelegramLogger

from utils.util import find_latest_epoch, save_plot, get_model_name


def main(config: dict):
    # region Setup
    seed_setup(config['setup']['seed'])

    run_mode: str = config['mode']
    result_dir_path: str = get_results_dir_path(config, run_mode)

    if run_mode == 'training':
        dirs_to_create = ('weights', 'samples', 'metrics')
        weights_path = join(result_dir_path, 'weights')
        metrics_path = join(result_dir_path, 'metrics')
    elif run_mode == 'experiments':
        dirs_to_create = tuple(experiment_functions_dict.keys())
        weights_path = join(get_results_dir_path(config, 'training'), 'weights')
        metrics_path = join(get_results_dir_path(config, 'training'), 'metrics')
    else:
        raise ValueError("mode should be `training` or `experiments`")

    results_dir_setup(result_dir_path, dirs_to_create)

    with open(join(result_dir_path, 'last_config.json'), mode='w') as f:
        json.dump(config, f)

    logging_setup(result_dir_path)
    log = logging.getLogger()

    log.info(f'Current mode {run_mode}')

    if config['telegram_logger']['enable']:
        tg_log = TelegramLogger.getLogger(config['telegram_logger'])

    device = cuda_setup(config['setup']['gpu_id'])
    log.info(f'Device variable: {device}')

    reconstruction_loss = ChamferLoss().to(device)
    full_model = FullModel(config['full_model']).to(device)
    full_model.apply(weights_init)

    optimizer = getattr(optim, config['training']['optimizer']['type'])  # class
    optimizer = optimizer(full_model.parameters(), **config['training']['optimizer']['hyperparams'])

    scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler']['type'])  # class
    scheduler = scheduler(optimizer, **config['training']['lr_scheduler']['hyperparams'])
    log.info(f'Model {get_model_name(config)} created')

    latest_epoch = find_latest_epoch(result_dir_path if run_mode == "training" else weights_path)

    log.info(f'Latest epoch found: {latest_epoch}')

    if latest_epoch > 0:
        if run_mode == "training":
            latest_epoch = restore_model_state(weights_path, metrics_path, config['setup']['gpu_id'], latest_epoch,
                                               "latest", full_model, optimizer, scheduler)
        elif run_mode == "experiments":
            latest_epoch = restore_model_state(weights_path, metrics_path, config['setup']['gpu_id'], latest_epoch,
                                               config['experiments']['epoch'], full_model)
        log.info(f'Restored epoch : {latest_epoch}')
    elif run_mode == "experiments":
        raise FileNotFoundError("no weights found at ", weights_path)
    # endregion Setup

    train_dataset, val_dataset_dict, test_dataset_dict = get_datasets(config['dataset'])

    log.info(f'Dataset loaded for classes: {[cat_name for cat_name in val_dataset_dict.keys()]}')

    if run_mode == 'training':
        samples_path = join(result_dir_path, 'samples')
        train_dataloader = DataLoader(train_dataset, pin_memory=True, **config['training']['dataloader']['train'])
        val_dataloaders_dict = {cat_name: DataLoader(cat_ds, pin_memory=True, **config['training']['dataloader']['val'])
                                for cat_name, cat_ds in val_dataset_dict.items()}
        if latest_epoch == 0:
            best_epoch_loss = np.Infinity
            train_losses = []
            val_losses = []
        else:
            train_losses, val_losses, best_epoch_loss = restore_metrics(metrics_path, latest_epoch)

        for epoch in range(latest_epoch + 1, config['training']['max_epoch'] + 1):
            start_epoch_time = datetime.now()
            log.debug("Epoch: %s" % epoch)

            full_model, optimizer, epoch_loss_all, epoch_loss_kld, epoch_loss_r, latest_partial, latest_gt, latest_rec \
                = train_epoch(epoch, full_model, optimizer, train_dataloader, device, reconstruction_loss,
                              config['training']['loss_coef'])
            scheduler.step()

            train_losses.append(np.array([epoch_loss_all, epoch_loss_r, epoch_loss_kld]))

            log_string = f'[{epoch}/{config["training"]["max_epoch"]}] ' \
                         f'Loss_ALL: {epoch_loss_all:.4f} ' \
                         f'Loss_R: {epoch_loss_r:.4f} ' \
                         f'Loss_E: {epoch_loss_kld:.4f} ' \
                         f'Time: {datetime.now() - start_epoch_time}'
            log.info(log_string)

            train_plots = []
            for k in range(min(5, latest_rec.shape[0])):
                train_plots.append(save_plot(latest_partial[k], epoch, k, samples_path, 'partial'))
                train_plots.append(save_plot(latest_rec[k], epoch, k, samples_path, 'reconstructed'))
                train_plots.append(save_plot(latest_gt[k].T, epoch, k, samples_path, 'gt'))

            if config['telegram_logger']['enable']:
                tg_log.log_images(train_plots[:9], log_string)

            epoch_val_losses, epoch_val_samples = val_epoch(epoch, full_model, device, val_dataloaders_dict,
                                                            val_dataset_dict.keys(), reconstruction_loss,
                                                            config['training']['loss_coef'])

            is_new_best = epoch_val_losses['total'][0] < best_epoch_loss

            if is_new_best:
                best_epoch_loss = epoch_val_losses['total'][0]

            val_losses.append(epoch_val_losses['total'])

            log_string = f'val results[{config["training"]["loss_coef"]}*our_cd]:\n'
            for k, v in epoch_val_losses.items():
                log_string += k + ': ' + str(v) + '\n'

            if is_new_best:
                log_string += "new best epoch"

            log.info(log_string)

            val_plots = []
            for cat_name, sample in epoch_val_samples.items():
                val_plots.append(save_plot(sample[0], epoch, cat_name, samples_path, 'val_partial'))
                val_plots.append(save_plot(sample[2], epoch, cat_name, samples_path, 'val_rec'))
                val_plots.append(save_plot(sample[1].T, epoch, cat_name, samples_path, 'val_gt'))

            if config['telegram_logger']['enable']:
                chosen_plot_idx = np.random.choice(np.arange(len(val_plots) / 3, dtype=np.int),
                                                   int(np.min([3, len(val_plots) / 3])), replace=False)
                plots_to_log = []
                for idx in chosen_plot_idx:
                    plots_to_log.extend(val_plots[3 * idx:3 * idx + 3])
                tg_log.log_images(plots_to_log, log_string)

            if (epoch % config['training']['state_save_frequency'] == 0 or is_new_best) \
                    and epoch > config['training'].get('min_save_epoch', 0):
                torch.save(full_model.state_dict(), join(weights_path, f'{epoch:05}_model.pth'))
                torch.save(optimizer.state_dict(), join(weights_path, f'{epoch:05}_O.pth'))
                torch.save(scheduler.state_dict(), join(weights_path, f'{epoch:05}_S.pth'))

                np.save(join(metrics_path, f'{epoch:05}_train'), np.array(train_losses))
                np.save(join(metrics_path, f'{epoch:05}_val'), np.array(val_losses))

                log_string = "Epoch: %s saved" % epoch
                log.debug(log_string)
                if config['telegram_logger']['enable']:
                    tg_log.log(log_string)

    elif run_mode == 'experiments':

        # from datasets.real_data import RealDataNPYDataset
        # test_dataset_dict = RealDataNPYDataset(root_dir="D:\\UJ\\bachelors\\3d-point-clouds-autocomplete\\data\\real_car_data")

        full_model.eval()

        with torch.no_grad():
            for experiment_name, experiment_dict in config['experiments']['settings'].items():
                if experiment_dict.pop('execute', False):
                    log.info(experiment_name)
                    experiment_functions_dict[experiment_name](full_model, device, test_dataset_dict, result_dir_path,
                                                               latest_epoch, **experiment_dict)

    exit(0)


if __name__ == '__main__':
    main(parse_config())
