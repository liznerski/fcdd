import numpy as np
import torch
from kornia import gaussian_blur2d
from scipy import signal
from skimage.filters import gaussian as gblur
from skimage.transform import rotate as im_rotate


def ceil(x):
    return int(np.ceil(x))


def floor(x):
    return int(np.floor(x))


def salt_and_pepper(size, p=0.5):
    return (torch.rand(size) < p).float()


def kernel_size_to_std(k):
    return np.log10(0.45*k + 1) + 0.25 if k < 20 else 10


def gkern(k, std=None):
    """Returns a 2D Gaussian kernel array with given kernel size k and std std"""
    if std is None:
        std = kernel_size_to_std(k)
    elif isinstance(std, str):
        std = float(std)
    if k % 2 == 0:
        # if kernel size is even, signal.gaussian returns center values sampled from gaussian at x=-1 and x=1
        # which is much less than 1.0 (depending on std). Instead, sample with kernel size k-1 and duplicate center
        # value, which is 1.0. Then divide whole signal by 2, because the duplicate results in a too high signal.
        gkern1d = signal.gaussian(k - 1, std=std).reshape(k - 1, 1)
        gkern1d = np.insert(gkern1d, (k - 1) // 2, gkern1d[(k - 1) // 2]) / 2
    else:
        gkern1d = signal.gaussian(k, std=std).reshape(k, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def confetti_noise(size, p=0.01, blobshaperange=((3, 3), (5, 5)), fillval=1.0, backval=0.0, ensureblob=True, awgn=0.0,
                   clamp=False, onlysquared=True, rotation=0, colored=False):
    assert len(size) == 4 or len(size) == 3, 'size must be n x c x h x w'
    if isinstance(blobshaperange[0], int) and isinstance(blobshaperange[1], int):
        blobshaperange = (blobshaperange, blobshaperange)
    assert len(blobshaperange) == 2
    assert len(blobshaperange[0]) == 2 and len(blobshaperange[1]) == 2
    assert not colored or len(size) == 4 and size[1] == 3
    out_size = size
    colors = []
    if len(size) == 3:
        size = (size[0], 1, size[1], size[2])  # add channel dimension
    else:
        size = tuple(size)  # Tensor(torch.size) -> tensor of shape size, Tensor((x, y)) -> Tensor with 2 elements x & y
    mask = (torch.rand((size[0], size[2], size[3])) < p).unsqueeze(1)
    while ensureblob and (mask.view(mask.size(0), -1).sum(1).min() == 0):
        idx = (mask.view(mask.size(0), -1).sum(1) == 0).nonzero().squeeze()
        s = idx.size(0) if len(idx.shape) > 0 else 1
        mask[idx] = (torch.rand((s, 1, size[2], size[3])) < p)
    res = torch.empty(size).fill_(backval)
    idx = mask.nonzero()

    all_shps = [
        (x, y) for x in range(blobshaperange[0][0], blobshaperange[1][0] + 1)
        for y in range(blobshaperange[0][1], blobshaperange[1][1] + 1) if not onlysquared or x == y
    ]
    picks = torch.FloatTensor(idx.size(0)).uniform_(0, len(all_shps)).int()
    nidx = []
    for n, blobshape in enumerate(all_shps):
        if (picks == n).sum() < 1:
            continue
        bhs = range(-(blobshape[0] // 2) if blobshape[0] % 2 != 0 else -(blobshape[0] // 2) + 1, blobshape[0] // 2 + 1)
        bws = range(-(blobshape[1] // 2) if blobshape[1] % 2 != 0 else -(blobshape[1] // 2) + 1, blobshape[1] // 2 + 1)
        extends = torch.stack([
            torch.zeros(len(bhs) * len(bws)).long(),
            torch.zeros(len(bhs) * len(bws)).long(),
            torch.arange(bhs.start, bhs.stop).repeat(len(bws)),
            torch.arange(bws.start, bws.stop).unsqueeze(1).repeat(1, len(bhs)).reshape(-1)
        ]).transpose(0, 1)
        nid = idx[picks == n].unsqueeze(1) + extends.unsqueeze(0)
        if colored:
            col = torch.randint(0, 256, (3, ))[:, None].repeat(1, nid.reshape(-1, nid.size(-1)).size(0)).byte()
            colors.append(col)
        nid = nid.reshape(-1, extends.size(1))
        nid = torch.max(torch.min(nid, torch.LongTensor(size) - 1), torch.LongTensor([0, 0, 0, 0]))
        nidx.append(nid)
    idx = torch.cat(nidx)
    shp = res[idx.transpose(0, 1).numpy()].shape
    if colored:
        colors = torch.cat(colors, dim=1)
        res[idx.transpose(0, 1).numpy()] = colors[0] + torch.randn(shp) * awgn
        res[(idx + torch.LongTensor((0, 1, 0, 0))).transpose(0, 1).numpy()] = colors[1] + torch.randn(shp) * awgn
        res[(idx + torch.LongTensor((0, 2, 0, 0))).transpose(0, 1).numpy()] = colors[2] + torch.randn(shp) * awgn
    else:
        res[idx.transpose(0, 1).numpy()] = torch.ones(shp) * fillval + torch.randn(shp) * awgn
        res = res[:, 0, :, :]
        if len(out_size) == 4:
            res = res.unsqueeze(1).repeat(1, out_size[1], 1, 1)
    if clamp:
        res = res.clamp(backval, fillval) if backval < fillval else res.clamp(fillval, backval)
    mask = mask[:, 0, :, :]
    if rotation > 0:
        idx = mask.nonzero()
        res = res.unsqueeze(1) if res.dim() != 4 else res
        res = res.transpose(1, 3).transpose(1, 2)
        for pick, blbctr in zip(picks, mask.nonzero()):
            rot = np.random.uniform(-rotation, rotation)
            p1, p2 = all_shps[pick]
            dims = (
                blbctr[0],
                slice(max(blbctr[1] - floor(0.75 * p1), 0), min(blbctr[1] + ceil(0.75 * p1), res.size(1) - 1)),
                slice(max(blbctr[2] - floor(0.75 * p2), 0), min(blbctr[2] + ceil(0.75 * p2), res.size(2) - 1)),
                ...
            )
            res[dims] = torch.from_numpy(
                im_rotate(res[dims], rot, order=0, cval=0, center=(blbctr[1]-dims[1].start, blbctr[2]-dims[2].start))
            )
        res = res.transpose(1, 2).transpose(1, 3)
        res = res.squeeze() if len(out_size) != 4 else res
    return res


def colorize_noise(img, color_min=(-255, -255, -255), color_max=(255, 255, 255), p=1):
    assert 0 <= p <= 1
    orig_img = img.clone()
    if len(set(color_min)) == 1 and len(set(color_max)) == 1:
        cmin, cmax = color_min[0], color_max[0]
        img[img != 0] = torch.randint(cmin, cmax+1, img[img != 0].shape).float()
    else:
        img = img.transpose(0, 1)
        for ch, (cmin, cmax) in enumerate(zip(color_min, color_max)):
            img[ch][img[ch] != 0] = torch.randint(cmin, cmax+1, img[ch][img[ch] != 0].shape).float()
    if p < 1:
        pmask = torch.rand(img[img != 0].shape) >= p
        tar = img[img != 0]
        tar[pmask] = orig_img[img != 0][pmask]
        img[img != 0] = tar
    return img


def smooth_noise(img, ksize, std, p=1.0, inplace=True):
    if not inplace:
        img = img.clone()
    ksize = ksize if ksize % 2 == 1 else ksize - 1
    picks = torch.from_numpy(np.random.binomial(1, p, size=img.size(0))).bool()
    if picks.sum() > 0:
        img[picks] = gaussian_blur2d(img[picks], (ksize, ) * 2, (std, ) * 2)
    return img


def gauscolor(size, p=0.7):
    assert len(size) == 4, 'size must be n x c x h x w'
    img = np.float32(np.random.binomial(n=1, p=p, size=size))
    img = img.transpose(0, 2, 3, 1)  # nchw -> nhwc
    for i, pic in enumerate(img):
        img[i] = gblur(pic, sigma=1.5, multichannel=False)
    img[img < 0.75] = 0.0
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img


def solid(size):
    assert len(size) == 4, 'size must be n x c x h x w'
    return torch.randint(0, 256, (size[:-2]))[:, :, None, None].repeat(1, 1, size[-2], size[-1]).byte()
