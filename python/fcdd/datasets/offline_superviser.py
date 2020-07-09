import torch


def noise(ds, generated_noise, norm, nom_class, train_set, gt=False):
    if gt:
        raise ValueError('No GT mode for pure noise available!')
    anom = generated_noise.clamp(0, 255).byte()
    data = torch.cat((norm, anom))
    targets = torch.cat(
        (torch.ones(norm.size(0)) * nom_class,
         torch.ones(anom.size(0)) * ds.outlier_classes[0])
    )
    train_set.data = data
    train_set.targets = targets
    return train_set


def malformed_normal(ds, generated_noise, norm, nom_class, train_set, gt=False, brightness_threshold=0.11*255):
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
         torch.ones(anom.size(0)) * ds.outlier_classes[0])
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
