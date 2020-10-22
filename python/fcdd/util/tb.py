import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from matplotlib.pyplot import get_cmap
import numpy as np

class TBLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalars(self, loss, lr, epoch):
        self.writer.add_scalar('Loss/train', loss, epoch)
        self.writer.add_scalar('LR', lr, epoch)
        self.writer.flush()

    def add_scalars(self, loss, loss_normal, loss_anomal, lr, epoch):
        self.writer.add_scalars('Loss/train', {
            'total': loss,
            'normal': loss_normal,
            'anomalous': loss_anomal
        }, epoch)

        self.writer.add_scalar('LR', lr, epoch)
        self.writer.flush()

    def add_images(self, inputs: torch.Tensor, gt_maps: (None, torch.Tensor), outputs: torch.Tensor,
                   normal: bool, epoch: int):
        main_tag = 'normal' if normal else 'anomalous'

        cmap = get_cmap('jet')
        outputs_new = []
        outputs = outputs.clone()

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        norm_ip(outputs, float(outputs.min()), float(outputs.max()))
        for img in outputs.squeeze(dim=1):
            outputs_new.append(cmap(img.detach().cpu().numpy())[:, :, :3])
        outputs = torch.tensor(outputs_new).permute(0, 3, 1, 2)

        for tag, imgs in zip(['inputs', 'gt_maps', 'outputs'], [inputs, gt_maps, outputs]):
            if imgs is not None:
                batch_size = imgs.size(0)
                nrow = int(np.sqrt(batch_size))
                grid = make_grid(imgs, nrow=nrow)
                self.writer.add_image(main_tag + '/' + tag, grid, epoch)
        self.writer.flush()

    def add_network(self, model: torch.nn.Module, input_to_model):
        self.writer.add_graph(model, input_to_model)
        self.writer.flush()

    def add_weight_histograms(self, model, epoch):
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                self.writer.add_histogram(name + '.weight', m.weight, epoch)
                if m.bias is not None:
                    self.writer.add_histogram(name + '.bias', m.bias, epoch)
        self.writer.flush()

    def close(self):
        self.writer.close()
