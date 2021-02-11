from typing import List

import torch
from torch.utils.data.dataset import Dataset


def noise(outlier_classes: List[int], generated_noise: torch.Tensor, norm: torch.Tensor,
          nom_class: int, train_set: Dataset, gt: bool = False) -> Dataset:
    """
    Creates a dataset based on the nominal classes of a given dataset and generated noise anomalies.
    :param outlier_classes: a list of all outlier class indices.
    :param generated_noise: torch tensor of noise images (might also be Outlier Exposure based noise) (n x c x h x w).
    :param norm: torch tensor of nominal images (n x c x h x w).
    :param nom_class: the index of the class that is considered nominal.
    :param train_set: some training dataset.
    :param gt: whether to provide ground-truth maps as well, atm not available!
    :return: a modified dataset, with training data consisting of nominal samples and artificial anomalies.
    """
    if gt:
        raise ValueError('No GT mode for pure noise available!')
    anom = generated_noise.clamp(0, 255).byte()
    data = torch.cat((norm, anom))
    targets = torch.cat(
        (torch.ones(norm.size(0)) * nom_class,
         torch.ones(anom.size(0)) * outlier_classes[0])
    )
    train_set.data = data
    train_set.targets = targets
    return train_set


def malformed_normal(outlier_classes: List[int], generated_noise: torch.Tensor, norm: torch.Tensor, nom_class: int,
                     train_set: Dataset, gt: bool = False, brightness_threshold: float = 0.11*255) -> Dataset:
    """
    Creates a dataset based on the nominal classes of a given dataset and generated noise anomalies.
    Unlike above, the noise images are not directly utilized as anomalies, but added to nominal samples to
    create malformed normal anomalies.
    :param outlier_classes: a list of all outlier class indices.
    :param generated_noise: torch tensor of noise images (might also be Outlier Exposure based noise) (n x c x h x w).
    :param norm: torch tensor of nominal images (n x c x h x w).
    :param nom_class: the index of the class that is considered nominal.
    :param train_set: some training dataset.
    :param gt: whether to provide ground-truth maps as well.
    :param brightness_threshold: if the average brightness (averaged over color channels) of a pixel exceeds this
        threshold, the noise image's pixel value is subtracted instead of added.
        This avoids adding brightness values to bright pixels, where approximately no effect is achieved at all.
    :return: a modified dataset, with training data consisting of nominal samples and artificial anomalies.
    """
    assert (norm.dim() == 4 or norm.dim() == 3) and generated_noise.shape == norm.shape
    norm_dim = norm.dim()
    if norm_dim == 3:
        norm, generated_noise = norm.unsqueeze(1), generated_noise.unsqueeze(1)  # assuming ch dim is skipped
    anom = norm.clone()

    # invert noise for bright regions (bright regions are considered being on average > brightness_threshold)
    generated_noise = generated_noise.int()
    bright_regions = norm.sum(1) > brightness_threshold * norm.shape[1]
    for ch in range(norm.shape[1]):
        gnch = generated_noise[:, ch]
        gnch[bright_regions] = gnch[bright_regions] * -1
        generated_noise[:, ch] = gnch

    anom = (anom.int() + generated_noise).clamp(0, 255).byte()
    data = torch.cat((norm, anom))
    targets = torch.cat(
        (torch.ones(norm.size(0)) * nom_class,
         torch.ones(anom.size(0)) * outlier_classes[0])
    )
    if norm_dim == 3:
        data = data.squeeze(1)
    train_set.data = data
    train_set.targets = targets
    if gt:
        gtmaps = torch.cat(
            (torch.zeros_like(norm)[:, 0].float(),  # 0 for nominal
             (norm != anom).max(1)[0].clone().float())  # 1 for anomalous
        )
        if norm_dim == 4:
            gtmaps = gtmaps.unsqueeze(1)
        return train_set, gtmaps
    else:
        return train_set
