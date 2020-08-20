from fcdd.training.bases import BaseADTrainer
from torch import Tensor


class HSCTrainer(BaseADTrainer):
    def loss(self, outs: Tensor, ins: Tensor, labels: Tensor, gtmaps: Tensor = None, reduce='mean'):
        """ computes the HSC loss """
        assert reduce in ['mean', 'none']
        if self.objective in ['hsc']:
            loss = self.__hsc_loss(outs, ins, labels, gtmaps, reduce)
        else:
            raise NotImplementedError('Objective {} is not defined yet.'.format(self.objective))
        return loss

    def __hsc_loss(self, outs: Tensor, ins: Tensor, labels: Tensor, gtmaps: Tensor, reduce: str):
        loss = (outs ** 2).sum(-1)
        loss = (loss + 1).sqrt() - 1
        if self.net.training:
            norm = loss[labels == 0]
            anom = (-(((1 - (-loss[labels == 1]).exp()) + 1e-31).log()))
            loss[(1 - labels).nonzero().squeeze()] = norm
            loss[labels.nonzero().squeeze()] = anom
        else:
            loss = loss
        return loss.mean() if reduce == 'mean' else loss

    def snapshot(self, epochs: int):
        self.logger.snapshot(self.net, self.opt, self.sched, epochs)

