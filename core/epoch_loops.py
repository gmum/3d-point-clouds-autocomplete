from tqdm import tqdm
import torch
import numpy as np

from model.full_model import FullModel


def train_epoch(epoch, full_model: FullModel, optimizer, loader, device, rec_loss_function, loss_coef=0.05):
    full_model.train()
    loss_all = 0.0
    loss_r = 0.0
    loss_kld = 0.0

    for i, point_data in tqdm(enumerate(loader, 1), total=len(loader)):
        optimizer.zero_grad()

        existing, missing, gt, _ = point_data

        existing = existing.to(device)
        missing = missing.to(device)
        gt = gt.to(device)

        reconstruction, logvar, mu = full_model(existing, missing, list(gt.shape), epoch, device)

        loss_r = torch.mean(
            loss_coef * rec_loss_function(gt, reconstruction.permute(0, 2, 1)))

        if full_model.mode.has_generativity():
            loss_kld = 0.5 * (torch.exp(logvar) + torch.square(mu) - 1 - logvar).sum()
            loss_kld = torch.div(loss_kld, existing.shape[0])
            loss_all = loss_r + loss_kld
            loss_kld += loss_kld.item()
        else:
            loss_all = loss_r
        loss_r += loss_r.item()
        loss_all += loss_all.item()

        loss_all.backward()
        optimizer.step()

    loss_all = loss_all / i
    loss_kld = loss_kld / i
    loss_r = loss_r / i

    return full_model, optimizer, loss_all, loss_kld, loss_r, \
           existing.detach().cpu().numpy(), gt.detach().cpu().numpy(), reconstruction.detach().cpu().numpy()


def val_epoch(epoch, full_model, device, loaders_dict, val_classes_names, loss_function, loss_coef=0.05):
    full_model.eval()

    val_losses = dict.fromkeys(val_classes_names)
    val_samples = dict.fromkeys(val_classes_names)

    with torch.no_grad():
        for cat_name, dl in loaders_dict.items():
            loss = 0.0
            for i, point_data in enumerate(dl, 1):
                existing, missing, gt, _ = point_data
                existing = existing.to(device)
                missing = missing.to(device)
                gt = gt.to(device)

                reconstruction = full_model(existing, missing, list(gt.shape), epoch, device)

                loss_our_cd = torch.mean(
                    loss_coef * loss_function(gt, reconstruction.permute(0, 2, 1)))

                loss += loss_our_cd.item()

            existing = existing.cpu().numpy()
            gt = gt.cpu().numpy()
            reconstruction = reconstruction.detach().cpu().numpy()

            val_samples[cat_name] = (existing[0], gt[0], reconstruction[0])
            val_losses[cat_name] = np.array([loss / i])

        total = np.zeros(1)
        for v in val_losses.values():
            total = np.add(total, v)
        val_losses['total'] = total / len(val_losses.keys())

    return val_losses, val_samples
