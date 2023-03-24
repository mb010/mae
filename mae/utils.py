import torch

from torch.optim.lr_scheduler import LambdaLR
from math import cos, pi

from torch.utils.data import DataLoader
from tqdm import tqdm


def CosineLinearWarmupScheduler(opt, warmup_epochs, max_epochs):
    """Cosine annealing with linear warmup.

    Args:
        opt (torch.optim.Optimizer): Optimizer to use.
        warmup_epochs (int): Number of epochs for warmup.
        total_epochs (int): Total number of epochs.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Learning rate scheduler.
    """
    # Reduce linear warmup epochs to account for 0th epoch
    warmup_epochs -= 1

    # Linear warmup schedule
    warmup_lr_schedule = lambda t: (t + 1) / warmup_epochs if t <= warmup_epochs else 1.0

    # Cosine annealing schedule
    cosine_lr_schedule = lambda t: 0.5 * (1 + cos(pi * t / max_epochs))

    # Combine schedules
    lr_schedule = lambda t: warmup_lr_schedule(t) * cosine_lr_schedule(t)

    return LambdaLR(opt, lr_schedule)


def embed_dataset(encoder, data, batch_size=400):
    """
    Embed dataset with given encoder
    """
    print("Embedding dataset...")
    train_loader = DataLoader(data, batch_size)
    device = next(encoder.parameters()).device
    feature_bank = []
    target_bank = []
    for data in tqdm(train_loader):
        # Load data and move to correct device
        x, y = data

        x_enc = encoder(x.to(device))

        feature_bank.append(x_enc.squeeze().detach().cpu())
        target_bank.append(y.to(device).detach().cpu())

    # Save full feature bank for validation epoch
    feature_bank = torch.cat(feature_bank)
    target_bank = torch.cat(target_bank)

    return feature_bank, target_bank
