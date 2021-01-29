from typing import Tuple

import numpy as np
import torch
from kornia import gaussian_blur2d
from scipy import signal
from skimage.transform import rotate as im_rotate


def ceil(x: float):
    return int(np.ceil(x))


def floor(x: float):
    return int(np.floor(x))


def salt_and_pepper(size: torch.Size, p=0.5):
    return (torch.rand(size) < p).float()


def kernel_size_to_std(k: int):
    """ Returns a standard deviation value for a Gaussian kernel based on its size """
    return np.log10(0.45*k + 1) + 0.25 if k < 32 else 10


def gkern(k: int, std: float = None):
    "" "Returns a 2D Gaussian kernel array with given kernel size k and std std """
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


def confetti_noise(size: torch.Size, p: float = 0.01,
                   blobshaperange: Tuple[Tuple[int, int], Tuple[int, int]] = ((3, 3), (5, 5)),
                   fillval: int = 255, backval: int = 0, ensureblob: bool = True, awgn: float = 0.0,
                   clamp: bool = False, onlysquared: bool = True, rotation: int = 0,
                   colorrange: Tuple[int, int] = None) -> torch.Tensor:
    """
    Generates "confetti" noise, as seen in the paper.
    The noise is based on sampling randomly many rectangles (in the following called blobs) at random positions.
    Additionally, all blobs are of random size (within some range), of random rotation, and of random color.
    The color is randomly chosen per blob, thus consistent within one blob.
    :param size: size of the overall noise image(s), should be (n x h x w) or (n x c x h x w), i.e.
        number of samples, channels, height, width. Blobs are grayscaled for (n x h x w) or c == 1.
    :param p: the probability of inserting a blob per pixel.
        The average number of blobs in the image is p * h * w.
    :param blobshaperange: limits the random size of the blobs. For ((h0, h1), (w0, w1)), all blobs' width
        is ensured to be in {w0, ..., w1}, and height to be in {h0, ..., h1}.
    :param fillval: if the color is not randomly chosen (see colored parameter), this sets the color of all blobs.
        This is also the maximum value used for clamping (see clamp parameter). Can be negative.
    :param backval: the background pixel value, i.e. the color of pixels in the noise image that are not part
         of a blob. Also used for clamping.
    :param ensureblob: whether to ensure that there is at least one blob per noise image.
    :param awgn: amount of additive white gaussian noise added to all blobs.
    :param clamp: whether to clamp all noise image to the pixel value range (backval, fillval).
    :param onlysquared: whether to restrict the blobs to be squares only.
    :param rotation: the maximum amount of rotation (in degrees)
    :param colorrange: the range of possible color values for each blob and channel.
        Defaults to None, where the blobs are not colored, but instead parameter fillval is used.
        First value can be negative.
    :return: torch tensor containing n noise images. Either (n x c x h x w) or (n x h x w), depending on size.
    """
    assert len(size) == 4 or len(size) == 3, 'size must be n x c x h x w'
    if isinstance(blobshaperange[0], int) and isinstance(blobshaperange[1], int):
        blobshaperange = (blobshaperange, blobshaperange)
    assert len(blobshaperange) == 2
    assert len(blobshaperange[0]) == 2 and len(blobshaperange[1]) == 2
    assert colorrange is None or len(size) == 4 and size[1] == 3
    out_size = size
    colors = []
    if len(size) == 3:
        size = (size[0], 1, size[1], size[2])  # add channel dimension
    else:
        size = tuple(size)  # Tensor(torch.size) -> tensor of shape size, Tensor((x, y)) -> Tensor with 2 elements x & y
    mask = (torch.rand((size[0], size[2], size[3])) < p).unsqueeze(1)  # mask[i, j, k] == 1 for center of blob
    while ensureblob and (mask.view(mask.size(0), -1).sum(1).min() == 0):
        idx = (mask.view(mask.size(0), -1).sum(1) == 0).nonzero().squeeze()
        s = idx.size(0) if len(idx.shape) > 0 else 1
        mask[idx] = (torch.rand((s, 1, size[2], size[3])) < p)
    res = torch.empty(size).fill_(backval).int()
    idx = mask.nonzero()  # [(idn, idz, idy, idx), ...] = indices of blob centers
    if idx.reshape(-1).size(0) == 0:
        return torch.zeros(out_size).int()

    all_shps = [
        (x, y) for x in range(blobshaperange[0][0], blobshaperange[1][0] + 1)
        for y in range(blobshaperange[0][1], blobshaperange[1][1] + 1) if not onlysquared or x == y
    ]
    picks = torch.FloatTensor(idx.size(0)).uniform_(0, len(all_shps)).int()  # for each blob center pick a shape
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
        if colorrange is not None:
            col = torch.randint(
                colorrange[0], colorrange[1], (3, )
            )[:, None].repeat(1, nid.reshape(-1, nid.size(-1)).size(0)).int()
            colors.append(col)
        nid = nid.reshape(-1, extends.size(1))
        nid = torch.max(torch.min(nid, torch.LongTensor(size) - 1), torch.LongTensor([0, 0, 0, 0]))
        nidx.append(nid)
    idx = torch.cat(nidx)  # all pixel indices that blobs cover, not only center indices
    shp = res[idx.transpose(0, 1).numpy()].shape
    if colorrange is not None:
        colors = torch.cat(colors, dim=1)
        gnoise = (torch.randn(3, *shp) * awgn).int() if awgn != 0 else (0, 0, 0)
        res[idx.transpose(0, 1).numpy()] = colors[0] + gnoise[0]
        res[(idx + torch.LongTensor((0, 1, 0, 0))).transpose(0, 1).numpy()] = colors[1] + gnoise[1]
        res[(idx + torch.LongTensor((0, 2, 0, 0))).transpose(0, 1).numpy()] = colors[2] + gnoise[2]
    else:
        gnoise = (torch.randn(shp) * awgn).int() if awgn != 0 else 0
        res[idx.transpose(0, 1).numpy()] = torch.ones(shp).int() * fillval + gnoise
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
                im_rotate(
                    res[dims].float(), rot, order=0, cval=0, center=(blbctr[1]-dims[1].start, blbctr[2]-dims[2].start),
                    clip=False
                )
            ).int()
        res = res.transpose(1, 2).transpose(1, 3)
        res = res.squeeze() if len(out_size) != 4 else res
    return res


def colorize_noise(img: torch.Tensor, color_min: Tuple[int, int, int] = (-255, -255, -255),
                   color_max: Tuple[int, int, int] = (255, 255, 255), p: float = 1) -> torch.Tensor:
    """
    Colorizes given noise images by asserting random color values to pixels that are not black (zero).
    :param img: torch tensor (n x c x h x w)
    :param color_min: limit the random color to be greater than this rgb value
    :param color_max: limit the random color to be less than this rgb value
    :param p: the chance to change the color of a pixel, on average changes p * h * w many pixels.
    :return: colorized images
    """
    assert 0 <= p <= 1
    orig_img = img.clone()
    if len(set(color_min)) == 1 and len(set(color_max)) == 1:
        cmin, cmax = color_min[0], color_max[0]
        img[img != 0] = torch.randint(cmin, cmax+1, img[img != 0].shape).type(img.dtype)
    else:
        img = img.transpose(0, 1)
        for ch, (cmin, cmax) in enumerate(zip(color_min, color_max)):
            img[ch][img[ch] != 0] = torch.randint(cmin, cmax+1, img[ch][img[ch] != 0].shape).type(img.dtype)
    if p < 1:
        pmask = torch.rand(img[img != 0].shape) >= p
        tar = img[img != 0]
        tar[pmask] = orig_img[img != 0][pmask]
        img[img != 0] = tar
    return img


def smooth_noise(img: torch.Tensor, ksize: int, std: float, p: float = 1.0, inplace: bool = True) -> torch.Tensor:
    """
    Smoothens (blurs) the given noise images with a Gaussian kernel.
    :param img: torch tensor (n x c x h x w).
    :param ksize: the kernel size used for the Gaussian kernel.
    :param std: the standard deviation used for the Gaussian kernel.
    :param p: the chance smoothen an image, on average smoothens p * n images.
    :param inplace: whether to apply the operation inplace.
    """
    if not inplace:
        img = img.clone()
    ksize = ksize if ksize % 2 == 1 else ksize - 1
    picks = torch.from_numpy(np.random.binomial(1, p, size=img.size(0))).bool()
    if picks.sum() > 0:
        img[picks] = gaussian_blur2d(img[picks].float(), (ksize, ) * 2, (std, ) * 2).int()
    return img


def solid(size: torch.Size):
    """ Returns noise images in form of solid colors, i.e. one randomly chosen color per image. """
    assert len(size) == 4, 'size must be n x c x h x w'
    return torch.randint(0, 256, (size[:-2]))[:, :, None, None].repeat(1, 1, size[-2], size[-1]).byte()
