import collections
import os.path as pt

import numpy as np
import torch
from fcdd.models.bases import SpatialCenterNet
from fcdd.training.bases import ObjectiveADTrainer


class CNNTrainer(ObjectiveADTrainer):
    def __init__(
            self, net, opt, sched, dataset_loaders, logger, device=torch.device('cuda:0'),
            objective='spatial_center', objective_params=None
    ):
        super(CNNTrainer, self).__init__(
            net, opt, sched, dataset_loaders, logger, device, objective, objective_params
        )

    def loss(self, outs, ins, labels, gtmaps=None, reduce='mean'):
        assert reduce in ['mean', 'none']
        if self.objective in ['hard_boundary', 'spatial_center']:
            loss = self.__hard_boundary_loss(outs, ins, labels, gtmaps, reduce)
        else:
            raise NotImplementedError('Objective {} is not defined yet.'.format(self.objective))
        return loss

    def __hard_boundary_loss(self, outs, ins, labels, gtmaps, reduce):
        loss = outs ** 2
        loss = (loss + 1).sqrt() - 1
        if gtmaps is None and len(set(labels.tolist())) > 1:
            loss = self.__supervised_loss(loss, labels)
        elif gtmaps is not None and isinstance(self.net, SpatialCenterNet):
            loss = self.__gt_loss(loss, gtmaps)
        return loss.mean() if reduce == 'mean' else loss

    def __supervised_loss(self, loss, labels):
        if self.net.training:
            loss = loss.reshape(labels.size(0), -1).mean(-1)
            norm = loss[labels == 0]
            anom = (-(((1 - (-loss[labels == 1]).exp()) + 1e-31).log()))
            loss[(1-labels).nonzero().squeeze()] = norm
            loss[labels.nonzero().squeeze()] = anom
        else:
            loss = loss
        if torch.isnan(loss).sum() > 0:
            print('WARNING: Error is nan!!!')
        return loss

    def __gt_loss(self, loss, gtmaps):
        if self.net.training:
            std = self.objective_params.get('gaus_std', None) if hasattr(self, "objective_params") else None
            loss = self.net.get_heatmap(loss, reception=True, std=std, cpu=False)[:, 0]
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
        if torch.isnan(loss).sum() > 0:
            print('WARNING: Error is nan!!!')
        return loss

    def interpretation_viz(self, labels, losses, ascores, imgs, outs, gtmaps=None, grads=None,
                           show_per_cls=20, nrow=None, name='heatmaps', qu=0.93, suffix='.', specific_idx=()):
        # Evaluation Picture with 4 rows. Each row splits into 4 subrows with input-output-heatmap-GT:
        # (1) 20 first nominal samples (2) 20 first anomalous samples
        # (3) 10 most nominal nominals - 10 most anomalous nominals
        # (4) 10 most nominal anomalies - 10 most anomalous anomalies
        tim, rh = self._compute_inter_viz(
            labels, ascores, imgs, outs, gtmaps, grads, show_per_cls, nrow,
            qu=qu, norm='semi_global', resdownsample=self.objective_params.get('resdown', 64),
        )
        show_per_cls = min(show_per_cls, min(collections.Counter(labels).values()))
        if show_per_cls % 2 != 0:
            show_per_cls -= 1
        self.logger.imsave(
            name + '_semi-global', tim, nrow=show_per_cls, scale_mode='none', rowheaders=rh, suffix=suffix
        )

        # Concise paper picture: 4 rows. Each row grows from most nominal to most anomalous (equidistant).
        # (1) input (2) pixelwise anomaly score (3) simple gradient (4) GT
        # Produce that picture for nominal and anomalous samples separately
        if 'train' not in name:
            res = 128
            rascores = self.reduce_ascore(ascores)
            k = show_per_cls // 3
            inpshp = imgs.shape
            for l in sorted(set(labels)):
                lid = set((torch.from_numpy(np.asarray(labels)) == l).nonzero().squeeze(-1).tolist())
                sort = [
                    i for i in np.argsort(rascores.detach().view(rascores.size(0), -1).sum(1)).tolist() if i in lid
                ]
                splits = np.array_split(sort, k)
                idx = [s[int(n / (k - 1) * len(s)) if n != len(splits) - 1 else -1] for n, s in enumerate(splits)]
                self.logger.logtxt(
                    'Interpretation visualization paper image {} indicies for label {}: {}'
                    .format('{}_paper_lbl{}'.format(name, l), l, idx)
                )
                self.create_paper_image(idx, name, inpshp, l, suffix, res, qu, imgs, ascores, grads, gtmaps)
                if specific_idx is not None and len(specific_idx) > 0:
                    self.create_paper_image(
                        specific_idx[l], name, inpshp, l,
                        pt.join(suffix, 'specific_viz_ids'),
                        res, qu, imgs, ascores, grads, gtmaps
                    )

    def create_paper_image(self, idx, name, inpshp, lbl, suffix, res, qu, imgs, ascores, grads, gtmaps):
        for norm in ['local', 'semi_global']:
            rows = []
            rows.append(self._make_heatmap(imgs[idx], inpshp, maxres=res))
            if self.objective != 'hard_boundary':
                rows.append(
                    self._make_heatmap(
                        ascores[idx], inpshp, maxres=res, qu=qu, colorize=True, norm=norm,
                        ref=ascores if norm == 'global' else None
                    )
                )
            if grads is not None:
                rows.append(
                    self._make_heatmap(
                        grads[idx], inpshp, True, res, qu=qu, colorize=True, norm=norm,
                        ref=grads if norm == 'global' else None
                    )
                )
            if gtmaps is not None:
                rows.append(self._make_heatmap(gtmaps[idx], inpshp, maxres=res, norm=None))
            tim = torch.cat(rows)
            imname = '{}_paper_{}_lbl{}'.format(name, norm, lbl)
            self.logger.single_save(imname, torch.stack(rows), suffix=pt.join('tims', suffix))
            self.logger.imsave(imname, tim, nrow=len(idx), scale_mode='none', suffix=suffix)

    def snapshot(self, epochs): 
        self.logger.snapshot(self.net, self.opt, self.sched, epochs)

