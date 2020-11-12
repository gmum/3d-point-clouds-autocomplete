import argparse
import json
import logging
import pickle
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from models import aae

from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging, set_seed, get_weights_dir
from utils.points import generate_points

cudnn.benchmark = True


def main(config):
    set_seed(config['seed'])

    results_dir = prepare_results_dir(config, config['arch'], 'experiments',
                                      dirs_to_create=['interpolations', 'sphere', 'points_interpolation',
                                                      'different_number_points', 'fixed', 'reconstruction',
                                                      'sphere_triangles', 'sphere_triangles_interpolation'])
    weights_path = get_weights_dir(config)
    epoch = find_latest_epoch(weights_path)

    if not epoch:
        print("Invalid 'weights_path' in configuration")
        exit(1)

    setup_logging(results_dir)
    global log
    log = logging.getLogger('aae')

    if not exists(join(results_dir, 'experiment_config.json')):
        with open(join(results_dir, 'experiment_config.json'), mode='w') as f:
            json.dump(config, f)

    device = cuda_setup(config['cuda'], config['gpu'])
    log.info(f'Device variable: {device}')
    if device.type == 'cuda':
        log.info(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'],
                                  is_sliced=True)
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.info("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']), len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                   num_workers=config["num_workers"], drop_last=True, pin_memory=True)

    #
    # Models
    #
    hyper_network = aae.HyperNetwork(config, device).to(device)
    encoder = aae.EncoderForRandomPoints(config).to(device)
    real_data_encoder = aae.EncoderForRealPoints(config).to(device)

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        # from utils.metrics import earth_mover_distance
        # reconstruction_loss = earth_mover_distance
        from losses.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    log.info("Weights for epoch: %s" % epoch)

    log.info("Loading weights...")
    hyper_network.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))
    encoder.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))
    real_data_encoder.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_ER.pth')))

    hyper_network.eval()
    encoder.eval()
    real_data_encoder.eval()

    total_loss_eg = 0.0
    total_loss_e = 0.0
    total_loss_kld = 0.0

    real_x = []
    remaining_x = []
    target_x = []

    samples_amount = np.max([int(e['amount']) for e in config['experiments'].values() if e['execute']])

    with torch.no_grad():
        for i, point_data in enumerate(points_dataloader, 1):

            real_X, remaining_X, target_X, _ = point_data

            real_X = real_X.to(device)
            remaining_X = remaining_X.to(device)
            target_X = target_X.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if real_X.size(-1) == 3:
                real_X.transpose_(real_X.dim() - 2, real_X.dim() - 1)

            if remaining_X.size(-1) == 3:
                remaining_X.transpose_(remaining_X.dim() - 2, remaining_X.dim() - 1)

            if target_X.size(-1) == 3:
                target_X.transpose_(target_X.dim() - 2, target_X.dim() - 1)

            if i - 1 < samples_amount:
                real_x.append(real_X)
                remaining_x.append(remaining_X)
                target_x.append(target_X)

            codes, mu, logvar = encoder(remaining_X)
            real_mu = real_data_encoder(real_X)

            target_networks_weights = hyper_network(torch.cat([codes, real_mu], 1))

            X_rec = torch.zeros(target_X.shape).to(device)

            for j, target_network_weights in enumerate(target_networks_weights):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)
                target_network_input = generate_points(config=config, epoch=epoch, size=(target_X.shape[2], target_X.shape[1]))
                X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

            loss_e = torch.mean(
                config['reconstruction_coef'] *
                reconstruction_loss(target_X.permute(0, 2, 1) + 0.5,
                                    X_rec.permute(0, 2, 1) + 0.5))

            loss_kld = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar).sum()

            loss_eg = loss_e + loss_kld
            total_loss_e += loss_e.item()
            total_loss_kld += loss_kld.item()
            total_loss_eg += loss_eg.item()

        log.info(
            f'Loss_ALL: {total_loss_eg / i:.4f} '
            f'Loss_R: {total_loss_e / i:.4f} '
            f'Loss_E: {total_loss_kld / i:.4f} '
        )

        real_x = torch.cat(real_x)
        remaining_x = torch.cat(remaining_x)
        target_x = torch.cat(target_x)

        for experiment_name, experiment_dict in config['experiments'].items():
            if experiment_dict.pop('execute', False):
                experiment_functions_dict[experiment_name](encoder, real_data_encoder, hyper_network, device, target_x,
                                                           remaining_x, real_x, results_dir, epoch, **experiment_dict)


def get_hyper_network_weights(remaining_x, real_x, rand_encoder, real_encoder, hyper_network):
    z_a, _, _ = rand_encoder(remaining_x)
    mu_r = real_encoder(real_x)
    return hyper_network(torch.cat([z_a, mu_r], 1))


def interpolation(rand_encoder, real_encoder, hyper_network, device, target_x, remaining_x, real_x, results_dir, epoch,
                  amount=5, transitions=10):
    log.info(f'Interpolations')

    for k in range(amount):
        x_ar = real_x[None, 2 * k, :, :]
        x_a = remaining_x[None, 2 * k, :, :]
        x_br = real_x[None, 2 * k + 1, :, :]
        x_b = remaining_x[None, 2 * k + 1, :, :]

        with torch.no_grad():
            z_a, mu_a, var_a = rand_encoder(x_a)
            z_b, mu_b, var_b = rand_encoder(x_b)
            mu_ar = real_encoder(x_ar)
            mu_br = real_encoder(x_br)

        for j, alpha in enumerate(np.linspace(0, 1, transitions)):
            z_int = (1 - alpha) * torch.cat([z_a, mu_ar], 1) + alpha * torch.cat([z_b, mu_br], 1)  # interpolate in the latent space
            weights_int = hyper_network(z_int)  # decode the interpolated sample

            target_network = aae.TargetNetwork(config, weights_int[0])
            target_network_input = generate_points(config=config, epoch=epoch, size=(
                real_x.shape[2] + remaining_x.shape[2],
                real_x.shape[1]))
            x_int = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()

            np.save(join(results_dir, 'interpolations', f'{k}_{j}_target_network_input'), np.array(target_network_input))
            np.save(join(results_dir, 'interpolations', f'{k}_{j}_interpolation'), np.array(x_int))

            fig = plot_3d_point_cloud(x_int[0], x_int[1], x_int[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'interpolations', f'{k}_{j}_interpolation.png'))
            plt.close(fig)


def interpolation_between_two_points(rand_encoder, real_encoder, hyper_network, device, target_x, remaining_x, real_x, results_dir, epoch,
                                     amount=30, image_points=1000, transitions=21):
    log.info("Interpolations between two points")

    weights_int = get_hyper_network_weights(remaining_x[:amount], real_x[:amount], rand_encoder, real_encoder,
                                            hyper_network)

    for k in range(amount):
        target_network = aae.TargetNetwork(config, weights_int[k])
        target_network_input = generate_points(config=config, epoch=epoch, size=(image_points, remaining_x.shape[1]))
        x_a = target_network_input[torch.argmin(target_network_input, dim=0)[2]][None, :]
        x_b = target_network_input[torch.argmax(target_network_input, dim=0)[2]][None, :]

        x_rec = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()
        x_int = torch.zeros(transitions, remaining_x.shape[1])
        for j, alpha in enumerate(np.linspace(0, 1, transitions)):
            z_int = (1 - alpha) * x_a + alpha * x_b  # interpolate point
            x_int[j] = target_network(z_int.to(device))

        x_int = torch.transpose(x_int, 0, 1).cpu().numpy()

        np.save(join(results_dir, 'points_interpolation', f'{k}_target_network_input'), np.array(target_network_input))
        np.save(join(results_dir, 'points_interpolation', f'{k}_reconstruction'), np.array(x_rec))
        np.save(join(results_dir, 'points_interpolation', f'{k}_points_interpolation'), np.array(x_int))

        fig = plot_3d_point_cloud(x_rec[0], x_rec[1], x_rec[2], in_u_sphere=True,
                                  show=False, x1=x_int[0], y1=x_int[1], z1=x_int[2])
        fig.savefig(join(results_dir, 'points_interpolation', f'{k}_points_interpolation.png'))
        plt.close(fig)


def reconstruction(rand_encoder, real_encoder, hyper_network, device, target_x, remaining_x, real_x, results_dir, epoch,
                   amount=5):
    log.info("Reconstruction")
    target_x = target_x[:amount].cpu().numpy()
    weights_rec = get_hyper_network_weights(remaining_x[:amount], real_x[:amount], rand_encoder, real_encoder, hyper_network)

    for k in range(amount):
        target_network = aae.TargetNetwork(config, weights_rec[k])
        target_network_input = generate_points(config=config, epoch=epoch, size=(target_x.shape[2], target_x.shape[1]))
        x_rec = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()

        np.save(join(results_dir, 'reconstruction', f'{k}_target_network_input'), np.array(target_network_input))
        np.save(join(results_dir, 'reconstruction', f'{k}_real'), np.array(target_x[k]))
        np.save(join(results_dir, 'reconstruction', f'{k}_reconstructed'), np.array(x_rec))

        fig = plot_3d_point_cloud(x_rec[0], x_rec[1], x_rec[2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'reconstruction', f'{k}_reconstructed.png'))
        plt.close(fig)

        fig = plot_3d_point_cloud(target_x[k][0], target_x[k][1], target_x[k][2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'reconstruction', f'{k}_real.png'))
        plt.close(fig)


def sphere(rand_encoder, real_encoder, hyper_network, device, target_x, remaining_x, real_x, results_dir, epoch, amount=10,
           image_points=10240, start=2.0, end=4.0, transitions=21):
    log.info("Sphere")
    target_x = target_x[:amount].cpu().numpy()
    weights_sphere = get_hyper_network_weights(remaining_x[:amount], real_x[:amount], rand_encoder, real_encoder, hyper_network)

    for k in range(amount):
        target_network = aae.TargetNetwork(config, weights_sphere[k])
        target_network_input = generate_points(config=config, epoch=epoch, size=(image_points, target_x.shape[1]),
                                               normalize_points=False)
        x_rec = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()

        np.save(join(results_dir, 'sphere', f'{k}_real'), np.array(target_x[k]))
        np.save(join(results_dir, 'sphere', f'{k}_point_cloud_before_normalization'),
                np.array(target_network_input))
        np.save(join(results_dir, 'sphere', f'{k}_reconstruction'), np.array(x_rec))

        target_network_input = target_network_input / torch.norm(target_network_input, dim=1).view(-1, 1)
        np.save(join(results_dir, 'sphere', f'{k}_point_cloud_after_normalization'),
                np.array(target_network_input))

        for coeff in np.linspace(start, end, num=transitions):
            coeff = round(coeff, 1)
            x_sphere = torch.transpose(target_network(target_network_input.to(device) * coeff), 0, 1).cpu().numpy()

            np.save(join(results_dir, 'sphere',
                         f'{k}_output_from_target_network_for_point_cloud_after_normalization_coefficient_{coeff}'),
                    np.array(x_sphere))

            fig = plot_3d_point_cloud(x_sphere[0], x_sphere[1], x_sphere[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'sphere', f'{k}_{coeff}_sphere.png'))
            plt.close(fig)

        fig = plot_3d_point_cloud(target_x[k][0], target_x[k][1], target_x[k][2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'sphere', f'{k}_real.png'))
        plt.close(fig)


def sphere_triangles(rand_encoder, real_encoder, hyper_network, device, target_x, remaining_x, real_x, results_dir, epoch, amount=10,
                     method='edge', depth=2, start=0.5, end=1.5, transitions=21):
    from utils.sphere_triangles import generate
    log.info("Sphere triangles")
    target_x = target_x[:amount].cpu().numpy()
    weights_sphere = get_hyper_network_weights(remaining_x[:amount], real_x[:amount], rand_encoder, real_encoder, hyper_network)

    for k in range(amount):
        target_network = aae.TargetNetwork(config, weights_sphere[k])
        target_network_input, triangulation = generate(method, depth)
        x_rec = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()

        np.save(join(results_dir, 'sphere_triangles', f'{k}_real'), np.array(target_x[k]))
        np.save(join(results_dir, 'sphere_triangles', f'{k}_point_cloud'), np.array(target_network_input))
        np.save(join(results_dir, 'sphere_triangles', f'{k}_reconstruction'), np.array(x_rec))

        with open(join(results_dir, 'sphere_triangles', f'{k}_triangulation.pickle'), 'wb') as triangulation_file:
            pickle.dump(triangulation, triangulation_file)

        fig = plot_3d_point_cloud(x_rec[0], x_rec[1], x_rec[2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'sphere_triangles', f'{k}_reconstructed.png'))
        plt.close(fig)

        for coefficient in np.linspace(start, end, num=transitions):
            coefficient = round(coefficient, 3)
            target_network_input_coefficient = target_network_input * coefficient
            x_sphere = torch.transpose(target_network(target_network_input_coefficient.to(device)), 0, 1).cpu().numpy()

            np.save(join(results_dir, 'sphere_triangles', f'{k}_point_cloud_coefficient_{coefficient}'),
                    np.array(target_network_input_coefficient))
            np.save(join(results_dir, 'sphere_triangles', f'{k}_reconstruction_coefficient_{coefficient}'), x_sphere)

            fig = plot_3d_point_cloud(x_sphere[0], x_sphere[1], x_sphere[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'sphere_triangles', f'{k}_{coefficient}_reconstructed.png'))
            plt.close(fig)

        fig = plot_3d_point_cloud(target_x[k][0], target_x[k][1], target_x[k][2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'sphere_triangles', f'{k}_real.png'))
        plt.close(fig)


def sphere_triangles_interpolation(rand_encoder, real_encoder, hyper_network, device, target_x, remaining_x, real_x, results_dir, epoch,
                                   amount=10, method='edge', depth=2, coefficient=1.5, transitions=20):
    from utils.sphere_triangles import generate
    log.info("Sphere triangles interpolation")

    for k in range(amount):
        x_ar = real_x[None, 2 * k, :, :]
        x_a = remaining_x[None, 2 * k, :, :]
        x_br = real_x[None, 2 * k + 1, :, :]
        x_b = remaining_x[None, 2 * k + 1, :, :]

        with torch.no_grad():
            z_a, mu_a, var_a = rand_encoder(x_a)
            z_b, mu_b, var_b = rand_encoder(x_b)
            mu_ar = real_encoder(x_ar)
            mu_br = real_encoder(x_br)

        for j, alpha in enumerate(np.linspace(0, 1, transitions)):
            z_int = (1 - alpha) * torch.cat([z_a, mu_ar], 1) + alpha * torch.cat([z_b, mu_br], 1)  # interpolate in the latent space
            weights_int = hyper_network(z_int)  # decode the interpolated sample

            target_network = aae.TargetNetwork(config, weights_int[0])
            target_network_input, triangulation = generate(method, depth)
            x_int = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()

            np.save(join(results_dir, 'sphere_triangles_interpolation', f'{k}_{j}_point_cloud'),
                    np.array(target_network_input))
            np.save(join(results_dir, 'sphere_triangles_interpolation', f'{k}_{j}_interpolation'), x_int)

            with open(join(results_dir, 'sphere_triangles_interpolation', f'{k}_{j}_triangulation.pickle'),
                      'wb') as triangulation_file:
                pickle.dump(triangulation, triangulation_file)

            fig = plot_3d_point_cloud(x_int[0], x_int[1], x_int[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'sphere_triangles_interpolation', f'{k}_{j}_interpolation.png'))
            plt.close(fig)

            target_network_input_coefficient = target_network_input * coefficient
            x_int_coeff = torch.transpose(target_network(target_network_input_coefficient.to(device)), 0, 1).cpu().numpy()

            np.save(join(results_dir,
                         'sphere_triangles_interpolation', f'{k}_{j}_point_cloud_coefficient_{coefficient}'),
                    np.array(target_network_input_coefficient))
            np.save(join(results_dir, 'sphere_triangles_interpolation',
                         f'{k}_{j}_interpolation_coefficient_{coefficient}'), x_int_coeff)

            fig = plot_3d_point_cloud(x_int_coeff[0], x_int_coeff[1], x_int_coeff[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'sphere_triangles_interpolation', f'{k}_{j}_{coefficient}_interpolation.png'))
            plt.close(fig)


def different_number_of_points(rand_encoder, real_encoder, hyper_network, device, target_x, remaining_x, real_x, results_dir, epoch,
                               amount=5, number_of_points_list=(10, 100, 1000, 2048, 10000)):
    log.info("Different number of points")
    target_x = target_x[:amount].cpu().numpy()
    remaining_x = remaining_x[:amount]

    weights_diff = get_hyper_network_weights(remaining_x[:amount], real_x[:amount], rand_encoder, real_encoder, hyper_network)

    for k in range(amount):
        np.save(join(results_dir, 'different_number_points', f'{k}_real'), np.array(target_x[k]))
        fig = plot_3d_point_cloud(target_x[k][0], target_x[k][1], target_x[k][2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'different_number_points', f'{k}_real.png'))
        plt.close(fig)

        target_network = aae.TargetNetwork(config, weights_diff[k])

        for number_of_points in number_of_points_list:
            target_network_input = generate_points(config=config, epoch=epoch, size=(number_of_points, target_x.shape[1]))
            x_diff = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()

            np.save(join(results_dir, 'different_number_points', f'{k}_target_network_input'),
                    np.array(target_network_input))
            np.save(join(results_dir, 'different_number_points', f'{k}_{number_of_points}'), np.array(x_diff))

            fig = plot_3d_point_cloud(x_diff[0], x_diff[1], x_diff[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'different_number_points', f'{k}_{number_of_points}.png'))
            plt.close(fig)


def fixed(rand_encoder, real_encoder, hyper_network, device, target_x, remaining_x, real_x, results_dir, epoch,
          amount=30, mean=0.0, std=0.015, noises_per_item=3, triangulation_config={'execute': False, 'method': 'edge', 'depth': 2}):
    log.info("Fixed")
    target_x = target_x[:amount].cpu().numpy()

    real_x = real_x[:amount]
    mu_ar = real_encoder(real_x)
    real_x = real_x.cpu().numpy()

    for i in range(noises_per_item):
        fixed_noise = torch.zeros(amount, target_x.shape[2] - real_x.shape[2]).normal_(mean=mean, std=std).to(device)

        weights_fixed = hyper_network(torch.cat([fixed_noise, mu_ar], 1))

        for j, weights in enumerate(weights_fixed):
            target_network = aae.TargetNetwork(config, weights).to(device)

            target_network_input = generate_points(config=config, epoch=epoch, size=(target_x.shape[2], target_x.shape[1]))
            fixed_rec = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()
            np.save(join(results_dir, 'fixed', f'{j}_{i}_target_network_input'), np.array(target_network_input))
            np.save(join(results_dir, 'fixed', f'{j}_{i}_fixed_reconstruction'), fixed_rec)

            fig = plot_3d_point_cloud(fixed_rec[0], fixed_rec[1], fixed_rec[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'fixed', f'{j}_{i}_fixed_reconstructed.png'))
            plt.close(fig)

            if triangulation_config['execute']:
                from utils.sphere_triangles import generate

                target_network_input, triangulation = generate(triangulation_config['method'], triangulation_config['depth'])

                with open(join(results_dir, 'fixed', f'{j}_{i}_triangulation.pickle'), 'wb') as triangulation_file:
                    pickle.dump(triangulation, triangulation_file)

                fixed_rec = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()
                np.save(join(results_dir, 'fixed', f'{j}_{i}_target_network_input_triangulation'),
                        np.array(target_network_input))
                np.save(join(results_dir, 'fixed', f'{j}_{i}_fixed_reconstruction_triangulation'), fixed_rec)

                fig = plot_3d_point_cloud(fixed_rec[0], fixed_rec[1], fixed_rec[2], in_u_sphere=True, show=False)
                fig.savefig(join(results_dir, 'fixed', f'{j}_{i}_fixed_reconstructed_triangulation.png'))
                plt.close(fig)

            np.save(join(results_dir, 'fixed', f'{j}_{i}_fixed_noise'), np.array(fixed_noise[j].cpu()))
            np.save(join(results_dir, 'fixed', f'{j}_{i}_real_x_part'), real_x[j])


experiment_functions_dict = {
    'interpolation': interpolation,
    'interpolation_between_two_points': interpolation_between_two_points,
    'reconstruction': reconstruction,
    'sphere': sphere,
    'sphere_triangles': sphere_triangles,
    'sphere_triangles_interpolation': sphere_triangles_interpolation,
    'different_number_of_points': different_number_of_points,
    'fixed': fixed,
}


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
