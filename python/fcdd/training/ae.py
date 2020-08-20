from fcdd.training.bases import BaseADTrainer
from torch import Tensor


class AETrainer(BaseADTrainer):
    def loss(self, outs: Tensor, ins: Tensor, labels: Tensor, gtmaps: Tensor = None, reduce='mean'):
        """ computes the reconstruction loss """
        assert reduce in ['mean', 'none']
        if self.net.training and len(set(labels.tolist())) > 1:
            self.logger.warning('AE training received more than one label. Is that on purpose?', unique=True)
        loss = (outs - ins) ** 2
        return loss.mean() if reduce == 'mean' else loss

    def snapshot(self, epochs: int):
        self.logger.snapshot(self.net, self.opt, self.sched, epochs)
