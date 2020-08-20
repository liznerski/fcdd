import torch
from fcdd.models.bases import FCDDNet
from fcdd.training.bases import BaseADTrainer
from torch import Tensor


class FCDDTrainer(BaseADTrainer):
    def loss(self, outs: Tensor, ins: Tensor, labels: Tensor, gtmaps: Tensor = None, reduce='mean'):
        """ computes the FCDD loss """
        assert reduce in ['mean', 'none']
        if self.objective in ['fcdd']:
            loss = self.__fcdd_loss(outs, ins, labels, gtmaps, reduce)
        else:
            raise NotImplementedError('Objective {} is not defined yet.'.format(self.objective))
        return loss

    def __fcdd_loss(self, outs: Tensor, ins: Tensor, labels: Tensor, gtmaps: Tensor, reduce: str):
        loss = outs ** 2
        loss = (loss + 1).sqrt() - 1
        if gtmaps is None and len(set(labels.tolist())) > 1:
            loss = self.__supervised_loss(loss, labels)
        elif gtmaps is not None and isinstance(self.net, FCDDNet):
            loss = self.__gt_loss(loss, gtmaps)
        return loss.mean() if reduce == 'mean' else loss

    def __supervised_loss(self, loss: Tensor, labels: Tensor):
        if self.net.training:
            loss = loss.reshape(labels.size(0), -1).mean(-1)
            norm = loss[labels == 0]
            anom = (-(((1 - (-loss[labels == 1]).exp()) + 1e-31).log()))
            loss[(1-labels).nonzero().squeeze()] = norm
            loss[labels.nonzero().squeeze()] = anom
        else:
            loss = loss
        return loss

    def __gt_loss(self, loss: Tensor, gtmaps: Tensor):
        if self.net.training:
            std = self.gauss_std
            loss = self.net.receptive_upsample(loss, reception=True, std=std, cpu=False)
            norm = (loss * (1 - gtmaps)).view(loss.size(0), -1).mean(-1)
            exclude_complete_nominal_samples = ((gtmaps == 1).view(gtmaps.size(0), -1).sum(-1) > 0)
            anom = torch.zeros_like(norm)
            if exclude_complete_nominal_samples.sum() > 0:
                a = (loss * gtmaps)[exclude_complete_nominal_samples]
                anom[exclude_complete_nominal_samples] = (
                    -(((1 - (-a.view(a.size(0), -1).mean(-1)).exp()) + 1e-31).log())
                )
            loss = norm + anom
        else:
            loss = loss
        return loss

    def snapshot(self, epochs: int):
        self.logger.snapshot(self.net, self.opt, self.sched, epochs)

