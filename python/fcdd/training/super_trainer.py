import torch
import numpy as np
import collections
import os.path as pt
from sklearn.metrics import roc_auc_score, roc_curve
from fcdd.training.bases import ObjectiveADTrainer
from fcdd.models import SpatialCenterNet
from fcdd.training.deep_svdd import CNNTrainer


class ADTrainer(object):
    """
    Trainer class for training the net, testing, logging and visualization
    :param net: net for training FCDD, i.e. minimize distance to some hypersphere with center c
    :param opt: opt for net
    :param sched: lr scheduler for net
    :param dataset_loaders: train and test loader
    :param logger: logger
    :param device: torch device
    :param objective: which center is to be taken. One of the following:
        - hard_boundary: hypersphere classifier
        - spatial_center: fcdd
        - autoencoder
    :param objective_params: dictionary with further parameters for objective, see OBJECTIVE_PARAMS HELP
    :param supervise_params: dictionary with further parameters for supervise modes, see SUPERVISE_PARAMS HELP
    :param quantile: quantile param for normlization used in viz of heatmaps
    """
    def __init__(
            self, net, opt, sched, dataset_loaders, logger, device=torch.device('cuda:0'),
            objective='spatial_center', objective_params=None, supervise_params=None, quantile=0.93
    ):
        # Objectives and specific objective initialization
        self.objective = objective
        self.objective_params = objective_params
        self.supervise_params = supervise_params
        if objective in ['spatial_center']:
            assert isinstance(net, SpatialCenterNet)

        if self.objective != 'autoencoder':
            self.trainer = CNNTrainer(
                net, opt, sched, dataset_loaders, logger, device, self.objective, self.objective_params,
            )
        else:
            self.trainer = AETrainer(
                net, opt, sched, dataset_loaders, logger, device, self.objective, self.objective_params
            )

        # Other
        self.logger = logger
        self.net = net
        self.opt = opt
        self.sched = sched
        self.device = device
        self.res = {}  # keys = {pt_roc, roc, gtmap_roc, prc, gtmap_prc}
        self.quantile = quantile

    def train(self, epochs, snap=None, acc_batches=1):
        """
        Trains the net for anomaly detection.
        :param epochs: no of epochs
        :param snap: path to snapshot, to load weights for model before any training.
        :param acc_batches: accumulate as many batches and do backprop only on accumulated once
        :return:
        """
        start = self.load(snap)

        try:
            self.trainer.train(epochs - start, acc_batches)
        finally:
            self.logger.save()
            self.logger.plot()
            self.trainer.snapshot(epochs)

    def test(self, specific_viz_ids=()):
        res = self.trainer.test(quantile=self.quantile, specific_viz_ids=specific_viz_ids)
        if res is not None:
            self.res.update(res)
        return self.res

    def load(self, path):
        epoch = 0
        if path is not None:
            epoch = self.trainer.load(path)
        return epoch


class AETrainer(ObjectiveADTrainer):
    def loss(self, outs, ins, labels, gtmaps=None, reduce='mean'):
        assert reduce in ['mean', 'none']
        if self.net.training and len(set(labels.tolist())) > 1:
            self.logger.warning('AE training received more than one label. Is that on purpose?', unique=True)
        loss = (outs - ins) ** 2
        return loss.mean() if reduce == 'mean' else loss

    def score(self, labels, ascores, imgs, outs, gtmaps=None, grads=None):
        rascores = self.reduce_pixelwise_ascore(ascores) if gtmaps is not None else None
        ascores = self.reduce_ascore(ascores).tolist()
        fpr, tpr, thresholds = roc_curve(labels, ascores)
        score = roc_auc_score(labels, ascores)
        res = {'roc': {'tpr': tpr, 'fpr': fpr, 'ths': thresholds, 'auc': score}}
        self.logger.single_plot(
            'roc_curve', tpr, fpr, xlabel='false positive rate', ylabel='true positive rate',
            legend=['auc={}'.format(score)]
        )
        self.logger.single_save('roc', res['roc'])
        self.logger.logtxt('##### TEST SCORE {} #####'.format(score), print=True)
        imgs = imgs[np.asarray(labels) == 0][np.argsort(np.asarray(ascores)[np.asarray(labels) == 0])]
        self.logger.imsave('most_normal', imgs[:32], scale_mode='each')
        self.logger.imsave('most_anomalous', imgs[-32:], scale_mode='each')
        if gtmaps is not None:
            self.logger.print('Computing GT test score...')
            ascores = rascores
            gtmaps = self.test_loader.dataset.dataset.get_original_gtmaps_normal_class()
            ascores = torch.nn.functional.interpolate(ascores, (gtmaps.shape[-2:]))
            flat_gtmaps, flat_ascores = gtmaps.reshape(-1).int().tolist(), ascores.reshape(-1).tolist()
            gtfpr, gttpr, gtthresholds = roc_curve(flat_gtmaps, flat_ascores)
            gt_roc_score = roc_auc_score(flat_gtmaps, flat_ascores)
            gtmap_roc_res = {'tpr': gttpr, 'fpr': gtfpr, 'ths': gtthresholds, 'auc': gt_roc_score}
            self.logger.single_plot(
                'roc_curve_gt_pixelwise', gttpr, gtfpr, xlabel='false positive rate', ylabel='true positive rate',
                legend=['auc={}'.format(gt_roc_score)]
            )
            self.logger.single_save(
                'roc_gt_pixelwise', gtmap_roc_res
            )
            self.logger.logtxt('##### GTMAP ROC TEST SCORE {} #####'.format(gt_roc_score), print=True)
            res['gtmap_roc'] = gtmap_roc_res

        return res

    def interpretation_viz(self, labels, losses, ascores, imgs, outs,
                           gtmaps=None, grads=None, show_per_cls=20, nrow=None, name='heatmaps', qu=0.93,
                           suffix='.', specific_idx=()):
        show_per_cls = min(show_per_cls, min(collections.Counter(labels).values()))
        if show_per_cls % 2 != 0:
            show_per_cls -= 1
        nrow = nrow or show_per_cls

        tim, rowheaders = self._compute_inter_viz(
            labels, losses, imgs, outs, gtmaps, None, show_per_cls, nrow, colorize_out=False, qu=qu,
            resdownsample=self.objective_params.get('resdown', 64)
        )
        self.logger.imsave(
            '{}'.format(name), tim, nrow=nrow, scale_mode='none', rowheaders=rowheaders,
        )

        # Concise paper picture: Each row grows from most nominal to most anomalous (equidistant).
        # (1) input (2) pixelwise anomaly score
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
            rows.append(
                self._make_heatmap(
                    ascores[idx], inpshp, maxres=res, qu=qu, colorize=True, norm=norm, blur=True, ae=True,
                    ref=ascores if norm == 'global' else None
                )
            )
            if gtmaps is not None:
                rows.append(self._make_heatmap(gtmaps[idx], inpshp, maxres=res, norm=None, ae=True))
            tim = torch.cat(rows)
            imname = 'ae_{}_paper_{}_lbl{}'.format(name, norm, lbl)
            self.logger.single_save(imname, torch.stack(rows), suffix=pt.join('tims', suffix))
            self.logger.imsave(imname, tim, nrow=len(idx), scale_mode='none', suffix=suffix)

    def snapshot(self, epochs):
        self.logger.snapshot(self.net, self.opt, self.sched, epochs)
