from fcdd.datasets.noise import salt_and_pepper, confetti_noise, colorize_noise, gauscolor, solid, smooth_noise
from fcdd.datasets.outlier_exposure.imagenet import OEImageNet, OEImageNet22k
from fcdd.datasets.outlier_exposure.cifar100 import OECifar100
from fcdd.datasets.outlier_exposure.emnist import OEEMNIST
import torch

MODES = [
    'gaussian', 'sandp', 'uniform_sandp', 'uniform', 'blob', 'huge_blob', 'rgb_blob', 'mixed_blob', 'mixed_blob_size',
    'mixed_blob_size_rot', 'mixed_blob_size_rot_large', 'mixed_blob_size_rot_small', 'mixed_blob_size_rot_many',
    'blob_size_rot_small', 'mixed_rec_size_rot', 'geo', 'geo_large', 'rgb_gaussian', 'solid', 'solid_blobs_many',
    'mixed_blob_size_rot_solid', 'smooth_blobs', 'mixed_smooth_blobs', 'confetti'
]


def generate_noise(noise_mode, size, params, logger=None, datadir=None):
    if noise_mode is not None:
        if noise_mode in ['gaussian']:
            generated_noise = (torch.randn(size) * 64)
        elif noise_mode in ['sandp']:
            generated_noise = (salt_and_pepper(size, 0.1) * 255)
        elif noise_mode in ['uniform_sandp']:
            generated_noise = (torch.rand(size) * salt_and_pepper(size, 0.5)).mul(255)
        elif noise_mode in ['uniform']:
            generated_noise = (torch.rand(size)).mul(255)
        elif noise_mode in ['blob']:
            generated_noise = confetti_noise(size, 0.002, (6, 6), fillval=255, clamp=False, awgn=0)
        elif noise_mode in ['huge_blob']:
            generated_noise = confetti_noise(size, 0.00001, (42, 42), fillval=255, clamp=False, awgn=0)
        elif noise_mode in ['rgb_blob']:
            generated_noise = confetti_noise(size, 0.0001, (22, 22), fillval=255, clamp=False, awgn=0)
            generated_noise = colorize_noise(generated_noise, (-255, -255, -255), (255, 255, 255))
        elif noise_mode in ['mixed_blob']:
            generated_noise_rgb = confetti_noise(size, 0.00001, (34, 34), fillval=255, clamp=False, awgn=0)
            generated_noise = confetti_noise(size, 0.00001, (34, 34), fillval=255, clamp=False, awgn=0)
            generated_noise_rgb = colorize_noise(generated_noise_rgb, (0, 0, 0), (255, 255, 255), p=1)
            generated_noise = generated_noise_rgb + generated_noise
        elif noise_mode in ['mixed_blob_size']:
            generated_noise_rgb = confetti_noise(size, 0.000015, ((4, 4), (50, 50)), fillval=255, clamp=False, awgn=0)
            generated_noise = confetti_noise(size, 0.000015, ((4, 4), (50, 50)), fillval=255, clamp=False, awgn=0)
            generated_noise_rgb = colorize_noise(generated_noise_rgb, (0, 0, 0), (255, 255, 255), p=1)
            generated_noise = generated_noise_rgb + generated_noise
        elif noise_mode in ['mixed_blob_size_rot']:
            generated_noise_rgb = confetti_noise(
                size, 0.000015, ((4, 4), (50, 50)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise = confetti_noise(
                size, 0.000015, ((4, 4), (50, 50)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise_rgb = colorize_noise(generated_noise_rgb, (0, 0, 0), (255, 255, 255), p=1)
            generated_noise = generated_noise_rgb + generated_noise
        elif noise_mode in ['mixed_blob_size_rot_large']:
            generated_noise_rgb = confetti_noise(
                size, 0.000008, ((8, 8), (100, 100)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise = confetti_noise(
                size, 0.000008, ((8, 8), (100, 100)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise_rgb = colorize_noise(generated_noise_rgb, (0, 0, 0), (255, 255, 255), p=1)
            generated_noise = generated_noise_rgb + generated_noise
        elif noise_mode in ['mixed_blob_size_rot_small']:
            generated_noise_rgb = confetti_noise(
                size, 0.004, ((2, 2), (8, 8)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise = confetti_noise(
                size, 0.004, ((2, 2), (8, 8)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise_rgb = colorize_noise(generated_noise_rgb, (0, 0, 0), (255, 255, 255), p=1)
            generated_noise = generated_noise_rgb + generated_noise
        elif noise_mode in ['mixed_blob_size_rot_many']:
            generated_noise_rgb = confetti_noise(
                size, 0.00006, ((4, 4), (70, 70)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise = confetti_noise(
                size, 0.00006, ((4, 4), (70, 70)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise_rgb = colorize_noise(generated_noise_rgb, (0, 0, 0), (255, 255, 255), p=1)
            generated_noise = generated_noise_rgb + generated_noise
        elif noise_mode in ['blob_size_rot_small']:
            generated_noise = confetti_noise(
                size, 0.003, ((2, 2), (7, 7)), fillval=255, clamp=False, awgn=0, rotation=45
            )
        elif noise_mode in ['mixed_rec_size_rot']:
            generated_noise_rgb = confetti_noise(
                size, 0.000015, ((4, 4), (50, 50)), fillval=255, clamp=False, awgn=0, rotation=45, onlysquared=False
            )
            generated_noise = confetti_noise(
                size, 0.000015, ((4, 4), (50, 50)), fillval=255, clamp=False, awgn=0, rotation=45, onlysquared=False
            )
            generated_noise_rgb = colorize_noise(generated_noise_rgb, (0, 0, 0), (255, 255, 255), p=1)
            generated_noise = generated_noise_rgb + generated_noise
        elif noise_mode in ['rgb_gaussian']:
            if len(size) != 4 or size[1] != 3:
                raise ValueError('Colored gaussian is colorized and needs an rgb image dataset!')
            generated_noise = gauscolor(size).mul(255).byte()
        elif noise_mode in ['imagenet']:
            generated_noise = next(iter(OEImageNet(
                size, limit_var=params.get('limit', None), root=datadir
            ).data_loader()))
        elif noise_mode in ['imagenet22k']:
            generated_noise = next(iter(
                OEImageNet22k(size, limit_var=params.get('limit', None), logger=logger, root=datadir).data_loader()
            ))
        elif noise_mode in ['cifar100']:
            generated_noise = next(iter(OECifar100(
                size, limit_var=params.get('limit', None), root=datadir
            ).data_loader()))
        elif noise_mode in ['emnist']:
            generated_noise = next(iter(OEEMNIST(
                size, limit_var=params.get('limit', None), root=datadir
            ).data_loader()))
        elif noise_mode in ['solid']:
            generated_noise = solid(size)
        elif noise_mode in ['solid_blobs_many']:
            generated_noise = confetti_noise(
                size, 0.004, ((2, 2), (8, 8)), fillval=255, clamp=False, awgn=0, rotation=45, colored=True
            )
        elif noise_mode in ['mixed_blob_size_rot_solid']:
            generated_noise_rgb = confetti_noise(
                size, 0.000018, ((4, 4), (50, 50)), fillval=255, clamp=False, awgn=0, rotation=45, colored=True
            )
            generated_noise = confetti_noise(
                size, 0.000012, ((4, 4), (50, 50)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise = generated_noise_rgb + generated_noise
        elif noise_mode in ['confetti', 'smooth_blobs']:
            generated_noise_rgb = confetti_noise(
                size, 0.000018, ((8, 8), (54, 54)), fillval=255, clamp=False, awgn=0, rotation=45, colored=True
            )
            generated_noise = confetti_noise(
                size, 0.000012, ((8, 8), (54, 54)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise = generated_noise_rgb + generated_noise
            generated_noise = smooth_noise(generated_noise, 25, 5, 1.0)
        elif noise_mode in ['mixed_smooth_blobs']:
            generated_noise_rgb = confetti_noise(
                size, 0.00002, ((5, 5), (50, 50)), fillval=255, clamp=False, awgn=0, rotation=45, colored=True
            )
            generated_noise = confetti_noise(
                size, 0.000014, ((5, 5), (50, 50)), fillval=255, clamp=False, awgn=0, rotation=45
            )
            generated_noise = generated_noise_rgb + generated_noise
            generated_noise = smooth_noise(generated_noise, 25, 6, 0.5)
        else:
            raise NotImplementedError('Supervise noise mode {} unknown.'.format(noise_mode))
        return generated_noise


if __name__ == '__main__':
    from fcdd.util import imsave
    import os.path as pt
    for mode in MODES:
        gray = ['gaussian', 'sandp', 'uniform_sandp', 'uniform']
        size = torch.Size([16, 3, 128, 128]) if not mode in gray else torch.Size([16, 128, 128])
        noise = generate_noise(mode, size, {})
        imsave(
            noise if not mode in gray else noise.unsqueeze(1).repeat(1, 3, 1, 1),
            pt.join(pt.dirname(__file__), '..', '..', 'data', 'noise_examples', '{}.png'.format(mode)),
            norm=True,
            padval=1,
            nrow=8
        )
