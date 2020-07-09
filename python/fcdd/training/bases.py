import collections
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from fcdd.datasets.bases import GTMapADDataset
from fcdd.datasets.noise import kernel_size_to_std
from fcdd.models import CNN28, CNN32, CNN224, SPACEN_CNN28_FCONV, SPACEN_CNN32_FCONV_S, SPACEN_CNN224_FCONV
from fcdd.util.logging import colorize as colorize_img
from kornia import gaussian_blur2d
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


def reorder(labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads, ds=None):
    if ds is not None and hasattr(ds, 'fixed_random_order'):
        assert gtmaps is None, \
            'original gtmaps loaded in score do not know order! Hence reordering is not allowed for GT datasets'
        o = ds.fixed_random_order
        labels = labels[o] if isinstance(labels, (torch.Tensor, np.ndarray)) else np.asarray(labels)[o].tolist()
        loss, anomaly_scores, imgs = loss[o], anomaly_scores[o], imgs[o]
        outputs, gtmaps = outputs[o], gtmaps
        grads = grads[o] if grads is not None else None
    return labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads


class BasicTrainer(object):
    def __init__(self, net, opt, sched, dataset_loaders, logger, device='cuda:0', **kwargs):
        self.net = net
        self.opt = opt
        self.sched = sched
        if len(dataset_loaders) == 2:
            self.train_loader, self.test_loader = dataset_loaders
            self.val_loader = None
        else:
            self.train_loader, self.val_loader, self.test_loader = dataset_loaders
        self.logger = logger
        self.device = device

    def train(self, epochs):
        self.net = self.net.to(self.device).train()
        for epoch in range(epochs):
            for n_batch, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss(outputs, inputs, labels)
                loss.backward()
                self.opt.step()
                self.logger.log(
                    epoch, n_batch, len(self.train_loader), loss,
                    infoprint='LR {} ID {}'.format(
                        ['{:.0e}'.format(p['lr']) for p in self.opt.param_groups],
                        str(self.__class__)[8:-2],
                    )
                )
            self.val(epoch)
            self.sched.step()
        return self.net

    def val(self, epoch):
        if self.val_loader is not None:
            with torch.no_grad():
                self.net = self.net.eval()
                all_loss = []
                for n_batch, data in enumerate(self.val_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, inputs, labels)
                    all_loss.append(loss.item())
                    self.logger.log_info(
                        {'val_err': loss}
                    )
                    self.logger.print(
                        'VAL EPOCH {:02d} NBAT {:04d}/{:04d} ERR {:01f} ID {}'.format(
                            epoch, n_batch, len(self.val_loader), loss,
                            str(self.__class__)[8:-2],
                        ), fps=False if n_batch == (len(self.val_loader) - 1) else True
                    )
                self.net = self.net.train()
            return self.net

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def loss(self, outs, ins, labels, reduce='mean'):
        pass

    @abstractmethod
    def score(self, labels, losses, imgs, outs):
        pass

    def load(self, path):
        snapshot = torch.load(path)
        net_state = snapshot.pop('net', None)
        opt_state = snapshot.pop('opt', None)
        sched_state = snapshot.pop('sched', None)
        epoch = snapshot.pop('epoch', None)
        if net_state is not None and self.net is not None:
            self.net.load_state_dict(net_state)
        if opt_state is not None and self.opt is not None:
            self.opt.load_state_dict(opt_state)
        if sched_state is not None and self.sched is not None:
            self.sched.load_state_dict(sched_state)
        print('Loaded {}{}{} with starting epoch {} for {}'.format(
            'net_state, ' if net_state else '', 'opt_state, ' if opt_state else '',
            'sched_state' if sched_state else '', epoch, str(self.__class__)[8:-2]
        ))
        return epoch


class ReceptiveNetTrainer(BasicTrainer):
    def __init__(self, net, opt, sched, dataset_loaders, logger, device='cuda:0', **kwargs):
        super().__init__(net, opt, sched, dataset_loaders, logger, device, **kwargs)

    def test(self):
        self.net = self.net.to(self.device).eval()
        all_labels, all_loss, all_imgs, all_outputs = [], [], [], []
        for n_batch, data in enumerate(self.test_loader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            if hasattr(self, 'objective_params') and self.objective_params.get('heatmaps', '') == 'grad':
                outputs = self.net(inputs)
                loss = self.loss(outputs, inputs, labels, reduce='none')
            else:
                with torch.no_grad():
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, inputs, labels, reduce='none')
            all_labels += labels.cpu().tolist()
            all_loss.append(loss.cpu())
            all_imgs.append(inputs.cpu())
            all_outputs.append(outputs.cpu())
            self.logger.print(
                'TEST {:04d}/{:04d} ID {}'.format(n_batch, len(self.test_loader), str(self.__class__)[8:-2]),
                fps=True
            )
        all_imgs = torch.cat(all_imgs)
        all_outputs = torch.cat(all_outputs)
        all_loss = torch.cat(all_loss)
        self.interpretation_viz(all_labels, all_loss, all_loss, all_imgs, all_outputs)
        return self.score(all_labels, all_loss, all_imgs, all_outputs)

    @abstractmethod
    def loss(self, outs, ins, labels, reduce='mean'):
        pass

    @abstractmethod
    def score(self, labels, losses, imgs, outs):
        pass

    def interpretation_viz(self, labels, losses, losses2, imgs, outs,
                           gtmaps=None, grads=None, show_per_cls=20, nrow=None, name='heatmaps'):
        tim, rowheaders = self._compute_inter_viz(
            labels, losses, imgs, outs, gtmaps, grads, show_per_cls, nrow
        )
        show_per_cls = min(show_per_cls, min(collections.Counter(labels).values()))
        if show_per_cls % 2 != 0:
            show_per_cls -= 1
        nrow = nrow or show_per_cls
        self.logger.imsave(
            name, tim, nrow=nrow, scale_mode='none', rowheaders=rowheaders,
        )

    def _compute_inter_viz(self, labels, losses, imgs, outs, gtmaps=None, grads=None,
                           show_per_cls=20, nrow=None, reception=True, further_idx=(),
                           colorize_out=False, blacken=(), resdownsample=224, qu=0.93, norm='semi_global',
                           further_only=False):
        self.logger.print('Computing interpretation visualization...')
        use_grad = hasattr(self, 'objective_params') and self.objective_params.get('heatmaps', '') in ['grad']
        use_blur = hasattr(self, 'objective_params') and self.objective_params.get('blur_heatmaps', '')
        std = self.objective_params.get('gaus_std', None) if hasattr(self, "objective_params") else None
        lbls = torch.IntTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        show_per_cls = min(show_per_cls, min(collections.Counter(labels.tolist()).values()))
        if further_idx is not None and len(further_idx) > 0 and not isinstance(further_idx, torch.Tensor):
            further_idx = torch.LongTensor(further_idx)
            assert further_idx.dim() == 2 and further_idx.shape[-1] == show_per_cls  # foo x show_per_cls
        assert not further_only or len(further_idx) > 0
        viz_outs = outs.dim() == imgs.dim()
        viz_loss = losses.dim() == imgs.dim() or use_grad
        viz_gtmaps = gtmaps is not None and gtmaps.dim() == imgs.dim()
        nrow = nrow or show_per_cls

        pics = []  # n_pics x c x h x w
        pgts = []  # n_pics x 1 x h x w
        pouts, plosses = [], []  # n_pics x heads x c x h x w
        rowheaders, tim = [], []

        if use_grad:
            losses = grads
            if losses is None:
                raise ValueError('Trainer shall use gradient based heatmaps, but no gradients are available!')

        # extract pics, outs, losses
        if not further_only:
            for l in sorted(set(lbls.tolist())):
                pics.append(imgs[lbls == l][:show_per_cls])
                if viz_outs:
                    pouts.append(outs[lbls == l][:show_per_cls])
                if viz_loss:
                    plosses.append(losses[lbls == l][:show_per_cls])
                if viz_gtmaps:
                    pgts.append(gtmaps[lbls == l][:show_per_cls])
        # extract pics, outs, losses for further indices
        if len(further_idx) > 0:
            for idx in further_idx:
                pics.append(imgs[idx])
                if viz_outs:
                    pouts.append(outs[idx])
                if viz_loss:
                    plosses.append(losses[idx])
                if viz_gtmaps:
                    pgts.append(gtmaps[idx])
        # transform to tensors of expected shape
        pics = torch.cat(pics)
        if len(pouts) > 0:
            pouts = torch.cat(pouts)
        if len(plosses) > 0:
            plosses = torch.cat(plosses)
        if len(pgts) > 0:
            pgts = torch.cat(pgts)

        # compute heatmaps for outs and losses, if viz required
        if use_grad:
            if viz_outs:
                # just upsample outputs --> n_pics x heads x 1 x h x w
                pouts = torch.nn.functional.interpolate(pouts, pics.shape[-2:]).unsqueeze(2)
                pouts = pouts.repeat(1, 1, pics.size(1), 1, 1)
                rowheaders.append('out')
            if viz_loss:
                plosses = plosses.unsqueeze(1)  # n_pics x heads x c x h x w
                rowheaders.extend(['err_{}'.format(i) for i in range(plosses.size(1))])
        elif hasattr(self.net, 'get_heatmap'):
            if viz_outs:
                pouts = self.net.get_heatmap(pouts, reception=reception, std=std)  # n_pics x heads x c x h x w
                rowheaders.extend(['out_{}'.format(i) for i in range(pouts.size(1))])
            if viz_loss:
                plosses = self.net.get_heatmap(plosses, reception=reception, std=std)  # n_pics x heads x c x h x w
                rowheaders.extend(['err_{}'.format(i) for i in range(plosses.size(1))])
        else:
            if viz_outs:
                assert pouts.shape == pics.shape, 'no heatmap available for net and out shape unequals inp shape'
                pouts = pouts.unsqueeze(1)  # n_pics x 1 x c x h x w
                rowheaders.append('out')
            if viz_loss:
                assert plosses.shape == pics.shape, 'no heatmap available for net and loss shape unequals inp shape'
                plosses = plosses.unsqueeze(1)  # n_pics x 1 x c x h x w
                rowheaders.append('err')

        # blur loss if requested
        if use_blur:
            r = self.net.reception['r'] if not hasattr(self.net, 'encoder') else self.net.encoder.reception['r']
            r = r - 1 if r % 2 == 0 else r
            plosses = gaussian_blur2d(plosses.squeeze(1), (r,) * 2, (std or kernel_size_to_std(r),) * 2).unsqueeze(1)

        # downsample to res x res if necessary, else picture get way too large
        if pics.shape[-1] > resdownsample:
            pics = F.interpolate(pics, (resdownsample, resdownsample), mode='nearest')
            if viz_outs:
                pouts = F.interpolate(
                    pouts.transpose(1, 2), (pouts.size(1), resdownsample, resdownsample), mode='nearest'
                ).transpose(1, 2)
            if viz_loss:
                plosses = F.interpolate(
                    plosses.transpose(1, 2), (plosses.size(1), resdownsample, resdownsample), mode='nearest'
                ).transpose(1, 2)
            if viz_gtmaps:
                pgts = F.interpolate(pgts, (resdownsample, resdownsample), mode='nearest')

        # normalize all but loss
        for img in [pics, pouts]:
            if img is None or len(img) == 0:
                continue
            img.sub_(
                img.reshape(img.size(0), -1).min(1)[0][(..., ) + (None, ) * (img.dim() - 1)]
            ).div_(img.reshape(img.size(0), -1).max(1)[0][(..., ) + (None, ) * (img.dim() - 1)])

        # define a function for all that is left to do
        def colorize_blacken_organize(plosses, pouts, pics, pgts, rowheaders, blacken, tim, colorize_out):
            # colorize
            plosses = plosses.mean(-3).unsqueeze(-3) if viz_loss else None  # mean over channels
            if viz_outs and pouts.size(-3) == 3:  # assume AE, visualize directly
                colorize_out = False
            else:
                pouts = pouts.mean(-3).unsqueeze(-3) if viz_outs else None  # mean over channels
            colorized = colorize_img(
                [pouts if colorize_out else None, plosses], norm=False
            )
            if colorize_out:
                pouts = colorized[0]
            elif viz_outs and pouts.size(-3) != 3:
                pouts = pouts.repeat(1, 1, 3, 1, 1)
            plosses = colorized[1]
            if pics.shape[1] == 1:
                pics = pics.repeat(1, 3, 1, 1)
            if viz_gtmaps:
                pgts = pgts.repeat(1, 3, 1, 1)
                rowheaders.append('gts')

            # make part of the samples black, if requested
            if len(blacken) > 0:
                blacken = torch.LongTensor(blacken)
                pics[blacken] = 0
                if viz_outs:
                    pouts[blacken] = 0
                if viz_loss:
                    plosses[blacken] = 0
                if viz_gtmaps:
                    pgts[blacken] = 0

            # organize all images s.t. we have always a row of input images followed by rows of heatmaps
            rows = int(np.ceil(pics.size(0) / nrow))
            for s in range(rows):
                tim.append(pics[s * nrow:s * nrow + nrow])
                if viz_outs:
                    tim.append(pouts[s * nrow:s * nrow + nrow].transpose(0, 1).reshape(-1, *pics.shape[1:]))
                if viz_loss:
                    tim.append(plosses[s * nrow:s * nrow + nrow].transpose(0, 1).reshape(-1, *pics.shape[1:]))
                if viz_gtmaps:
                    tim.append(pgts[s * nrow:s * nrow + nrow])
                tim.append(torch.zeros_like(pics[s * nrow:s * nrow + nrow]))
            tim = torch.cat(tim)

            rowheaders = (['inp', *rowheaders, ''] * rows)[:-1]
            return tim, rowheaders

        if norm == 'global':
            return colorize_blacken_organize(
                self.__global_norm(plosses, qu, ref=losses), pouts, pics, pgts, rowheaders, blacken, tim, colorize_out
            )
        elif norm == 'semi_global':
            return colorize_blacken_organize(
                self.__global_norm(plosses, qu), pouts, pics, pgts, rowheaders, blacken, tim, colorize_out
            )
        elif norm == 'local':
            return colorize_blacken_organize(
                self.__local_norm(plosses, qu), pouts, pics, pgts, rowheaders, blacken, tim, colorize_out
            )
        elif norm == 'all':
            return [
                colorize_blacken_organize(
                    self.__global_norm(plosses, qu), deepcopy(pouts), deepcopy(pics),
                    deepcopy(pgts), deepcopy(rowheaders), blacken, deepcopy(tim), colorize_out
                ),
                colorize_blacken_organize(
                    self.__local_norm(plosses, qu), deepcopy(pouts), deepcopy(pics),
                    deepcopy(pgts), deepcopy(rowheaders), blacken, deepcopy(tim), colorize_out
                ),
                colorize_blacken_organize(
                    self.__global_norm(plosses, qu, ref=losses), deepcopy(pouts), deepcopy(pics),
                    deepcopy(pgts), deepcopy(rowheaders), blacken, deepcopy(tim), colorize_out
                )
            ]
        else:
            raise NotImplementedError('Quantile normalization type {} not known for heatmap losses.'.format(norm))

    def _make_heatmap(self, ascores, input_shape, blur=False, maxres=64, norm='local', qu=1,
                      colorize=False, ae=False, ref=None, cmap='jet', inplace=True):
        if not inplace:
            ascores = deepcopy(ascores)
        assert ascores.dim() == len(input_shape) == 4
        std = self.objective_params.get('gaus_std', None) if hasattr(self, "objective_params") else None
        if ascores.shape[2:] != input_shape[2:]:
            assert hasattr(self.net, 'get_heatmap'), 'ascore shape is not input shape, but no get_heatmap available!'
            ascores = self.net.get_heatmap(ascores, reception=True, std=std)[:, 0]  # n_pics x heads x c x h x w
        if blur:
            assert hasattr(self.net, 'reception'), 'blurring heatmaps desired, but no net has no receptive field!'
            if ae:
                enc = self.net.encoder
                if any(isinstance(m, torch.nn.Linear) for m in list(self.net.modules())[1:]):
                    if isinstance(enc, CNN28):
                        r = SPACEN_CNN28_FCONV(49, self.net.in_shape).reception['r']
                    elif isinstance(enc, CNN32):
                        r = SPACEN_CNN32_FCONV_S(enc.final_dim, self.net.in_shape).reception['r']
                    elif isinstance(enc, CNN224):
                        r = SPACEN_CNN224_FCONV(enc.final_dim, self.net.in_shape).reception['r']
                r = r - 1 if r % 2 == 0 else r
            elif any(isinstance(m, torch.nn.Linear) for m in list(self.net.modules())[1:]):  # HSC
                assert isinstance(self.net, (CNN28, CNN32, CNN224))
                if isinstance(self.net, CNN28):
                    r = SPACEN_CNN28_FCONV(self.net.final_dim, self.net.in_shape).reception['r']
                elif isinstance(self.net, CNN32):
                    r = SPACEN_CNN32_FCONV_S(self.net.final_dim, self.net.in_shape).reception['r']
                elif isinstance(self.net, CNN224):
                    r = SPACEN_CNN224_FCONV(self.net.final_dim, self.net.in_shape).reception['r']
                r = r - 1 if r % 2 == 0 else r
            else:
                r = self.net.reception['r'] - 1 if self.net.reception['r'] % 2 == 0 else self.net.reception['r']
            std = std or kernel_size_to_std(r)
            ascores = gaussian_blur2d(ascores, (r,) * 2, (std,) * 2)
        if maxres < max(ascores.shape[2:]):
            assert ascores.shape[-2] == ascores.shape[-1], 'ascores are not squares (rather rectangles)'
            ascores = F.interpolate(ascores, (maxres, maxres), mode='nearest')
        ascores = [ascores]
        if norm is not None:
            apply_norm = {
                'local': self.__local_norm, 'global': self.__global_norm, 'semi_global': self.__global_norm
            }
            if norm == 'all':
                for key in ['local', 'global', 'semi_global']:
                    ascores.append(apply_norm[key](ascores[0], qu, ref if key == 'global' else None))
                ascores = ascores[1:]
            else:
                ascores = [apply_norm[norm](ascores[0], qu, ref if norm == 'global' else None)]

        def apply_colorize(ascores):
            ascores = ascores.mean(1).unsqueeze(1)
            colorized = colorize_img([ascores, ], norm=False, cmap=cmap)[0]
            return colorized

        if colorize:
            ascores = tuple(apply_colorize(a) for a in ascores)
        else:
            ascores = tuple(a.repeat(1, 3, 1, 1) if a.size(1) == 1 else a for a in ascores)

        return ascores if len(ascores) > 1 else ascores[0]

    @staticmethod
    def __global_norm(plosses, qu, ref=None):
        ref = ref if ref is not None else plosses
        plosses.sub_(ref.min())
        quantile = ref.view(-1).kthvalue(int(qu * ref.view(-1).size(0)))[0]  # qu% are below that
        plosses.div_(quantile)  # (1 - qu)% values will end up being out of scale ( > 1)
        plosses = plosses.clamp(0, 1)  # clamp those
        return plosses

    @staticmethod
    def __local_norm(plosses, qu, ref=None):
        plosses.sub_(plosses.view(plosses.size(0), -1).min(1)[0][(...,) + (None,) * (plosses.dim() - 1)])
        quantile = plosses.view(plosses.size(0), -1).kthvalue(
            int(qu * plosses.view(plosses.size(0), -1).size(1)), dim=1
        )[0]  # qu% are below that
        plosses.div_(quantile[(...,) + (None,) * (plosses.dim() - 1)])
        plosses = plosses.clamp(0, 1)  # clamp those
        return plosses


class BasicADTrainer(ReceptiveNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_res = None

    def train(self, epochs, acc_batches=1):
        assert 0 < acc_batches and isinstance(acc_batches, int)
        self.net = self.net.to(self.device).train()
        for epoch in range(epochs):
            acc_data, acc_counter = [], 1
            for n_batch, data in enumerate(self.train_loader):

                # accumulate batches and do backpropagation only on accumulated ones
                # reason: small batch-size == fast data loading, but small batch-size == unstable gradient
                if acc_counter < acc_batches and n_batch < len(self.train_loader) - 1:
                    acc_data.append(data)
                    acc_counter += 1
                    continue
                elif acc_batches > 1:
                    acc_data.append(data)
                    data = [torch.cat(d) for d in zip(*acc_data)]
                    acc_data, acc_counter = [], 1

                if isinstance(self.train_loader.dataset, GTMapADDataset):
                    inputs, labels, gtmaps = data
                    gtmaps = gtmaps.to(self.device)
                else:
                    inputs, labels = data
                    gtmaps = None
                inputs = inputs.to(self.device)
                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss(outputs, inputs, labels, gtmaps)
                loss.backward()
                self.opt.step()
                with torch.no_grad():
                    info = {}
                    if len(set(labels.tolist())) > 1:
                        swloss = self.loss(outputs, inputs, labels, gtmaps, reduce='none')
                        swloss = swloss.view(swloss.size(0), -1).mean(-1)
                        info = {'err_normal': swloss[labels == 0].mean(), 'err_anomalous': swloss[labels != 0].mean()}
                    self.logger.log(
                        epoch, n_batch, len(self.train_loader), loss,
                        infoprint='LR {} ID {}{}'.format(
                            ['{:.0e}'.format(p['lr']) for p in self.opt.param_groups],
                            str(self.__class__)[8:-2],
                            ' NCLS {}'.format(self.train_loader.dataset.normal_classes)
                            if hasattr(self.train_loader.dataset, 'normal_classes') else ''
                        ),
                        info=info
                    )
            self.val(epoch)
            self.sched.step()
        return self.net

    def test(self, quantile=0.93, specific_viz_ids=()):
        self.net = self.net.to(self.device).eval()

        self.logger.print('Test train data...', fps=False)
        labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = self._gather_data(
            self.train_loader
        )
        self.interpretation_viz(
            labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads,
            name='train_heatmaps', qu=quantile
        )

        self.logger.print('Test test data...', fps=False)
        labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = self._gather_data(
            self.test_loader,  # gather_all=True
        )
        labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = reorder(
            labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads, ds=self.test_loader.dataset
        )
        self.interpretation_viz(
            labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads,
            qu=quantile, specific_idx=specific_viz_ids
        )

        with torch.no_grad():
            sc = self.score(labels, anomaly_scores, imgs, outputs, gtmaps, grads)
        return sc

    def _gather_data(self, loader, gather_all=False):
        all_labels, all_loss, all_anomaly_scores, all_imgs, all_outputs = [], [], [], [], []
        all_gtmaps, all_grads = [], []
        for n_batch, data in enumerate(loader):
            if isinstance(loader.dataset, GTMapADDataset):
                inputs, labels, gtmaps = data
                all_gtmaps.append(gtmaps)
            else:
                inputs, labels = data
            bk_inputs = inputs.detach().clone()
            inputs = inputs.to(self.device)
            if gather_all:
                outputs, loss, anomaly_score, _ = self._regular_forward(inputs, labels)
                inputs = bk_inputs.clone().to(self.device)
                _, _, _, grads = self._grad_forward(inputs, labels)
            elif hasattr(self, 'objective_params') and self.objective_params.get('heatmaps', '') == 'grad':
                outputs, loss, anomaly_score, grads = self._grad_forward(inputs, labels)
            else:
                outputs, loss, anomaly_score, grads = self._regular_forward(inputs, labels)
            all_labels += labels.detach().cpu().tolist()
            all_loss.append(loss.detach().cpu())
            all_anomaly_scores.append(anomaly_score.detach().cpu())
            all_imgs.append(inputs.detach().cpu())
            all_outputs.append(outputs.detach().cpu())
            if grads is not None:
                all_grads.append(grads.detach().cpu())
            self.logger.print(
                'TEST {:04d}/{:04d} ID {}{}'.format(
                    n_batch, len(loader), str(self.__class__)[8:-2],
                    ' NCLS {}'.format(loader.dataset.normal_classes)
                    if hasattr(loader.dataset, 'normal_classes') else ''
                ),
                fps=True
            )
        all_imgs = torch.cat(all_imgs)
        all_outputs = torch.cat(all_outputs)
        all_gtmaps = torch.cat(all_gtmaps) if len(all_gtmaps) > 0 else None
        all_loss = torch.cat(all_loss)
        all_anomaly_scores = torch.cat(all_anomaly_scores)
        all_grads = torch.cat(all_grads) if len(all_grads) > 0 else None
        ret = (
            all_labels, all_loss, all_anomaly_scores, all_imgs, all_outputs, all_gtmaps,
            all_grads
        )
        return ret

    def _regular_forward(self, inputs, labels):
        with torch.no_grad():
            outputs = self.net(inputs)
            loss = self.loss(outputs, inputs, labels, reduce='none')
            anomaly_score = self.anomaly_score(loss)
            grads = None
        return outputs, loss, anomaly_score, grads

    def _grad_forward(self, inputs, labels):
        inputs.requires_grad = True
        outputs = self.net(inputs)
        loss = self.loss(outputs, inputs, labels, reduce='none')
        anomaly_score = self.anomaly_score(loss)
        grads = self.net.get_grad_heatmap(loss, outputs, inputs)
        inputs.requires_grad = False
        self.opt.zero_grad()
        return outputs, loss, anomaly_score, grads

    def score(self, labels, ascores, imgs, outs, gtmaps=None, grads=None, suffix='.'):
        # Logging
        self.logger.print('Computing test score...')
        if torch.isnan(ascores).sum() > 0:
            self.logger.logtxt('Could not compute test scores, since adscores contain nan values!!!', True)
            return None
        red_ascores = self.reduce_ascore(ascores).tolist()
        std = self.objective_params.get('gaus_std', None) if hasattr(self, "objective_params") else None

        # Overall ROC for sample-wise anomaly detection
        fpr, tpr, thresholds = roc_curve(labels, red_ascores)
        roc_score = roc_auc_score(labels, red_ascores)
        roc_res = {'tpr': tpr, 'fpr': fpr, 'ths': thresholds, 'auc': roc_score}
        self.logger.single_plot(
            'roc_curve', tpr, fpr, xlabel='false positive rate', ylabel='true positive rate',
            legend=['auc={}'.format(roc_score)], suffix=suffix
        )
        self.logger.single_save('roc', roc_res, suffix=suffix)
        self.logger.logtxt('##### ROC TEST SCORE {} #####'.format(roc_score), print=True)

        # Overall PRC for sample-wise anomaly detection
        prec, recall, thresholds = precision_recall_curve(labels, red_ascores)
        prc_score = auc(recall[1:], prec[:-1])
        prc_res = {'prec': prec, 'recall': recall, 'ths': thresholds, 'auc': prc_score}
        self.logger.single_plot(
            'prc_curve', prec, recall, xlabel='recall', ylabel='precision',
            legend=['auc={}'.format(prc_score)], suffix=suffix
        )
        self.logger.single_save('prc', prc_res, suffix=suffix)
        self.logger.logtxt('##### PRC TEST SCORE {} #####'.format(prc_score), print=True)

        # Most nominal and anomalous images
        imgs = imgs[np.asarray(labels) == 0][np.argsort(np.asarray(red_ascores)[np.asarray(labels) == 0])]
        self.logger.imsave('most_normal', imgs[:32], scale_mode='each', suffix=suffix)
        self.logger.imsave('most_anomalous', imgs[-32:], scale_mode='each', suffix=suffix)

        # GTMAPS pixel-wise anomaly detection
        gtmap_roc_res, gtmap_prc_res = None, None
        use_grads = hasattr(self, 'objective_params') and self.objective_params.get('heatmaps', '') in ['grad']
        if gtmaps is not None:
            self.logger.print('Computing GT test score...')
            ascores = self.reduce_pixelwise_ascore(ascores) if not use_grads else grads
            gtmaps = self.test_loader.dataset.dataset.get_original_gtmaps_normal_class()
            if not use_grads:
                ascores = self.net.get_heatmap(ascores, 1, std=std)[:, :, 0]
            ascores = torch.nn.functional.interpolate(ascores, (gtmaps.shape[-2:]))
            flat_gtmaps, flat_ascores = gtmaps.reshape(-1).int().tolist(), ascores.reshape(-1).tolist()

            # ROC
            gtfpr, gttpr, gtthresholds = roc_curve(flat_gtmaps, flat_ascores)
            gt_roc_score = roc_auc_score(flat_gtmaps, flat_ascores)
            gtmap_roc_res = {'tpr': gttpr, 'fpr': gtfpr, 'ths': gtthresholds, 'auc': gt_roc_score}
            self.logger.single_plot(
                'roc_curve_gt_pixelwise', gttpr, gtfpr, xlabel='false positive rate', ylabel='true positive rate',
                legend=['auc={}'.format(gt_roc_score)], suffix=suffix
            )
            self.logger.single_save(
                'roc_gt_pixelwise', gtmap_roc_res, suffix=suffix
            )
            self.logger.logtxt('##### GTMAP ROC TEST SCORE {} #####'.format(gt_roc_score), print=True)

            # PRC
            gtprec, gtrecall, gtthresholds = precision_recall_curve(flat_gtmaps, flat_ascores)
            gt_prc_score = auc(gtrecall[1:], gtprec[:-1])
            gtmap_prc_res = {'prec': gtprec, 'recall': gtrecall, 'ths': gtthresholds, 'auc': gt_prc_score}
            self.logger.single_plot(
                'prc_curve_gt_pixelwise', gtprec, gtrecall, xlabel='recall', ylabel='precision',
                legend=['auc={}'.format(gt_prc_score)], suffix=suffix
            )
            self.logger.single_save(
                'prc_gt_pixelwise', gtmap_prc_res, suffix=suffix
            )
            self.logger.logtxt('##### GTMAP PRC TEST SCORE {} #####'.format(gt_prc_score), print=True)

        return {'roc': roc_res, 'prc': prc_res, 'gtmap_roc': gtmap_roc_res, 'gtmap_prc': gtmap_prc_res}

    @abstractmethod
    def loss(self, outs, ins, labels, gtmaps=None, reduce='mean'):
        pass

    def anomaly_score(self, loss):
        return loss

    def reduce_ascore(self, ascore):
        return ascore.reshape(ascore.size(0), -1).mean(1)

    def reduce_pixelwise_ascore(self, ascore):
        return ascore.mean(1).unsqueeze(1)

    def interpretation_viz(self, labels, losses, ascores, imgs, outs, gtmaps=None, grads=None,
                           show_per_cls=20, nrow=None, name='heatmaps', qu=0.93, specific_idx=()):
        tim, rowheaders = self._compute_inter_viz(
            labels, ascores, imgs, outs, gtmaps, grads, show_per_cls, nrow, qu=qu
        )
        show_per_cls = min(show_per_cls, min(collections.Counter(labels).values()))
        if show_per_cls % 2 != 0:
            show_per_cls -= 1
        nrow = nrow or show_per_cls

        self.logger.imsave(
            name, tim, nrow=show_per_cls, scale_mode='none', rowheaders=rowheaders
        )

        tim, rowheaders = self._compute_inter_viz(
            labels, ascores, imgs, outs, gtmaps, grads, show_per_cls, nrow, False, qu=qu
        )
        self.logger.imsave(
            '{}_upsampled'.format(name), tim, nrow=show_per_cls, scale_mode='none', rowheaders=rowheaders
        )

    def _compute_inter_viz(self, labels, ascores, imgs, outs, gtmaps=None, grads=None,
                           show_per_cls=20, nrow=None, reception=True, further_idx=(),
                           colorize_out=False, margin_case=True, blacken=(), resdownsample=128,
                           qu=0.93, norm='semi_global', further_only=False,):
        lbls = torch.IntTensor(labels)
        show_per_cls = min(show_per_cls, min(collections.Counter(labels).values()))
        if show_per_cls % 2 != 0:
            show_per_cls -= 1
        if margin_case:
            # compute margin case, i.e. k most normal images and k most anomalous images
            rascores = self.reduce_ascore(ascores)
            k = show_per_cls // 2
            further_idx = list(further_idx)
            for l in sorted(set(lbls.tolist())):
                lid = set((lbls == l).nonzero().squeeze(-1).tolist())
                sort = [
                    i for i in np.argsort(rascores.detach().view(rascores.size(0), -1).sum(1)).tolist() if i in lid
                ]
                further_idx.append([*sort[:k], *sort[-k:]])
        return super()._compute_inter_viz(
            lbls, ascores, imgs, outs, gtmaps, grads, show_per_cls, nrow, reception, further_idx,
            colorize_out, blacken, resdownsample, qu, norm, further_only
        )


class ObjectiveADTrainer(BasicADTrainer):
    def __init__(self, net, opt, sched, dataset_loaders, logger, device='cuda:0',
                 objective='hard_boundary', objective_params=None, **kwargs):
        super().__init__(net, opt, sched, dataset_loaders, logger, device, **kwargs)
        self.objective = objective
        self.objective_params = objective_params or {}

    @abstractmethod
    def loss(self, outs, ins, labels, gtmaps=None, reduce='mean'):
        pass
