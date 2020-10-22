import collections
import os.path as pt
from abc import abstractmethod, ABC
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from fcdd.datasets.bases import GTMapADDataset
from fcdd.datasets.noise import kernel_size_to_std
from fcdd.models.bases import BaseNet, ReceptiveNet
from fcdd.util.logging import colorize as colorize_img, Logger
from fcdd.util.tb import TBLogger
from fcdd.training import balance_labels
from kornia import gaussian_blur2d
from sklearn.metrics import roc_auc_score, roc_curve


def reorder(labels: [int], loss: Tensor, anomaly_scores: Tensor, imgs: Tensor, outputs: Tensor, gtmaps: Tensor,
            grads: Tensor, ds: Dataset = None) -> ([int], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
    """ returns all inputs in an identical new order if the dataset offers a predefined (random) order """
    if ds is not None and hasattr(ds, 'fixed_random_order'):
        assert gtmaps is None, \
            'original gtmaps loaded in score do not know order! Hence reordering is not allowed for GT datasets'
        o = ds.fixed_random_order
        labels = labels[o] if isinstance(labels, (Tensor, np.ndarray)) else np.asarray(labels)[o].tolist()
        loss, anomaly_scores, imgs = loss[o], anomaly_scores[o], imgs[o]
        outputs, gtmaps = outputs[o], gtmaps
        grads = grads[o] if grads is not None else None
    return labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads


class BaseTrainer(ABC):
    def __init__(self, net: BaseNet, opt: Optimizer, sched: _LRScheduler, dataset_loaders: (DataLoader, DataLoader),
                 logger: Logger, tb_logger: TBLogger, device='cuda:0', **kwargs):
        """
        Base class for trainers, defines a simple train method and a method to load snapshots.
        At least the abstract loss method needs to be implemented, as it is used in the
        train method, but not yet defined. This base trainer does not yet support AD.
        :param net: some neural network instance
        :param opt: optimizer.
        :param sched: learning rate scheduler.
        :param dataset_loaders:
        :param logger: some logger.
        :param tb_logger: Tensorboard logger.
        :param device: some torch device, either cpu or gpu.
        :param kwargs: ...
        """
        self.net = net
        self.opt = opt
        self.sched = sched
        self.train_loader, self.test_loader = dataset_loaders
        self.logger = logger
        self.tb_logger = tb_logger
        self.device = device

    def train(self, epochs: int) -> BaseNet:
        """ Does epochs many full iteration of the data loader and trains the network with the data using self.loss """
        self.net = self.net.to(self.device).train()
        with torch.no_grad():
            inputs, _ = next(iter(self.train_loader))
            inputs = inputs.to(self.device)
            self.tb_logger.add_network(self.net, inputs)

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
            self.sched.step()
            mask = labels == 0
            self.tb_logger.add_scalars(loss, self.opt.param_groups[0]['lr'], epoch)
            self.tb_logger.add_weight_histograms(self.net, epoch)
            self.tb_logger.add_images(inputs[mask], None, outputs[mask], True, epoch)
            self.tb_logger.add_images(inputs[~mask], None, outputs[~mask], False, epoch)
        self.tb_logger.close()
        return self.net

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def loss(self, outs: Tensor, ins: Tensor, labels: Tensor, reduce='mean'):
        pass

    @abstractmethod
    def score(self, labels: Tensor, losses: Tensor, ins: Tensor, outs: Tensor):
        pass

    def load(self, path: str) -> int:
        """ Loads a snapshot of the training state, including network weights """
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


class BaseADTrainer(BaseTrainer):
    def __init__(self, net: BaseNet, opt: Optimizer, sched: _LRScheduler, dataset_loaders: (DataLoader, DataLoader),
                 logger: Logger, tb_logger: TBLogger, objective: str, gauss_std: float, quantile: float, resdown: int, blur_heatmaps=False,
                 device='cuda:0', **kwargs):
        """
        Anomaly detection trainer that defines a test phase where scores are computed and heatmaps are generated.
        The train method is modified to be able to handle ground-truth maps.
        :param gauss_std: a constant value for the standard deviation of the Gaussian kernel used for upsampling and
            blurring, the default value is determined by :func:`fcdd.datasets.noise.kernel_size_to_std`.
        :param quantile: the quantile that is used to normalize the generated heatmap images.
        :param resdown: the maximum resolution of logged images, images will be downsampled if necessary.
        :param blur_heatmaps: whether to blur heatmaps.
        """
        super().__init__(net, opt, sched, dataset_loaders, logger, tb_logger, device, **kwargs)
        self.objective = objective
        self.gauss_std = gauss_std
        self.quantile = quantile
        self.resdown = resdown
        self.blur_heatmaps = blur_heatmaps

    @abstractmethod
    def loss(self, outs: Tensor, ins: Tensor, labels: Tensor, gtmaps: Tensor = None, reduce='mean'):
        pass

    def anomaly_score(self, loss: Tensor) -> Tensor:
        """ This assumes the loss is already the anomaly score. If this is not the case, reimplement the method! """
        return loss

    def reduce_ascore(self, ascore: Tensor) -> Tensor:
        """ Reduces the anomaly score to be a score per image (detection). """
        return ascore.reshape(ascore.size(0), -1).mean(1)

    def reduce_pixelwise_ascore(self, ascore: Tensor) -> Tensor:
        """ Reduces the anomaly score to be a score per pixel (explanation). """
        return ascore.mean(1).unsqueeze(1)

    def train(self, epochs: int, acc_batches=1) -> BaseNet:
        """
        In addition to the base class this train method support ground-truth maps, logs losses for
        nominal and anomalous samples separately, and introduces another parameter to
        accumulate batches for faster data loading.
        :param epochs: number of full data loader iterations to train.
        :param acc_batches: To speed up data loading, this determines the number of batches that are accumulated
            before forwarded through the network. For instance, acc_batches=2 iterates the data loader two times,
            concatenates the batches, and passes this to the network. This has no impact on the performance
            if the batch size is reduced accordingly (e.g. one half in this example), but can decrease training time.
        :return: the trained network
        """
        assert 0 < acc_batches and isinstance(acc_batches, int)
        self.net = self.net.to(self.device).train()
        with torch.no_grad():
            inputs, _ = next(iter(self.train_loader))
            inputs = inputs.to(self.device)
            self.tb_logger.add_network(self.net, inputs)

        for epoch in range(epochs):
            acc_data, acc_counter = [], 1
            for n_batch, data in enumerate(self.train_loader):
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
                        swloss = swloss.reshape(swloss.size(0), -1).mean(-1)
                        info = {'err_normal': swloss[labels == 0].mean(),
                                'err_anomalous': swloss[labels != 0].mean()}
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

            self.tb_logger.add_weight_histograms(self.net, epoch)
            if len(set(labels.tolist())) > 1:
                mask = labels == 0
                self.tb_logger.add_scalars(loss, info['err_normal'], info['err_anomalous'],
                                        self.opt.param_groups[0]['lr'], epoch)
                self.tb_logger.add_images(inputs[mask], gtmaps[mask] if gtmaps is not None else None,
                                          outputs[mask], True, epoch)
                self.tb_logger.add_images(inputs[~mask], gtmaps[~mask] if gtmaps is not None else None,
                                          outputs[~mask], False, epoch)

            self.sched.step()
        self.tb_logger.close()
        return self.net

    def test(self, specific_viz_ids: ([int], [int]) = ()) -> dict:
        """
        Does a full iteration of the data loaders, remembers all data (i.e. inputs, labels, outputs, loss),
        and computes scores and heatmaps with it. Scores and heatmaps are computed for both, the training
        and the test data. For each, one heatmap picture is generated that contains (row-wise):
            -   The first 20 nominal samples (label == 0, if nominal_label==1 this shows anomalies instead).
            -   The first 20 anomalous samples (label == 1, if nominal_label==1 this shows nominal samples instead).
                The :func:`reorder` takes care that the first anomalous test samples are not all from the same class.
            -   The 10 most nominal rated samples from the nominal set on the left and
                the 10 most anomalous rated samples from the nominal set on the right.
            -   The 10 most nominal rated samples from the anomalous set on the left and
                the 10 most anomalous  rated samples from the anomalous set on the right.
        Additionally, for the test set only, four heatmap pictures are generated that show six samples with
        increasing anomaly score from left to right. Thereby the leftmost heatmap shows the most nominal rated example
        and the rightmost sample the most anomalous rated one. There are two heatmaps for the anomalous set and
        two heatmaps for the nominal set. Both with either local normalization -- i.e. each heatmap is normalized
        w.r.t itself only, there is a complete red and complete blue pixel in each heatmap -- or semi-global
        normalization -- each heatmap is normalized w.r.t. to all heatmaps shown in the picture.
        These four heatmap pictures are also stored as tensors in a 'tim' subdirectory for later usage.
        The score computes AUC values and complete ROC curves for detection. It also computes explanation ROC curves
        if ground-truth maps are available.

        :param specific_viz_ids: in addition to the heatmaps generated above, this also generates heatmaps
            for specific sample indices. The first element of specific_viz_ids is for nominal samples
            and the second for anomalous ones. The resulting heatmaps are stored in a `specific_viz_ids` subdirectory.
        :return: A dictionary of ROC results, each ROC result is again represented by a dictionary of the form: {
                'tpr': [], 'fpr': [], 'ths': [], 'auc': int, ...
            }.
        """
        self.net = self.net.to(self.device).eval()

        self.logger.print('Test training data...', fps=False)
        labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = self._gather_data(
            self.train_loader
        )
        self.heatmap_generation(
            labels, anomaly_scores, imgs, gtmaps, grads,
            name='train_heatmaps',
        )

        self.logger.print('Test test data...', fps=False)
        labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = self._gather_data(
            self.test_loader,
        )
        labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads = reorder(
            labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads, ds=self.test_loader.dataset
        )
        self.heatmap_generation(
            labels, anomaly_scores, imgs, gtmaps, grads,
            specific_idx=specific_viz_ids
        )

        with torch.no_grad():
            sc = self.score(labels, anomaly_scores, imgs, outputs, gtmaps, grads)
        return sc

    def _gather_data(self, loader: DataLoader,
                     gather_all=False) -> ([int], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
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
            elif self.objective == 'hsc':
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

    def _regular_forward(self, inputs: Tensor, labels: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        with torch.no_grad():
            outputs = self.net(inputs)
            loss = self.loss(outputs, inputs, labels, reduce='none')
            anomaly_score = self.anomaly_score(loss)
            grads = None
        return outputs, loss, anomaly_score, grads

    def _grad_forward(self, inputs: Tensor, labels: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        inputs.requires_grad = True
        outputs = self.net(inputs)
        loss = self.loss(outputs, inputs, labels, reduce='none')
        anomaly_score = self.anomaly_score(loss)
        grads = self.net.get_grad_heatmap(loss, inputs)
        inputs.requires_grad = False
        self.opt.zero_grad()
        return outputs, loss, anomaly_score, grads

    def score(self, labels: [int], ascores: Tensor, imgs: Tensor, outs: Tensor, gtmaps: Tensor = None,
              grads: Tensor = None, subdir='.') -> dict:
        """
        Computes the ROC curves and the AUC for detection performance.
        Also computes those for the explanation performance if ground-truth maps are available.
        :param labels: labels
        :param ascores: anomaly scores
        :param imgs: input images
        :param outs: outputs of the neural network
        :param gtmaps: ground-truth maps (can be None)
        :param grads: gradients of anomaly scores w.r.t. inputs (can be None)
        :param subdir: subdirectory to store the data in (plots and numbers)
        :return:  A dictionary of ROC results, each ROC result is again represented by a dictionary of the form: {
                'tpr': [], 'fpr': [], 'ths': [], 'auc': int, ...
            }.
        """
        # Logging
        self.logger.print('Computing test score...')
        if torch.isnan(ascores).sum() > 0:
            self.logger.logtxt('Could not compute test scores, since anomaly scores contain nan values!!!', True)
            return None
        red_ascores = self.reduce_ascore(ascores).tolist()
        std = self.gauss_std

        # Overall ROC for sample-wise anomaly detection
        fpr, tpr, thresholds = roc_curve(labels, red_ascores)
        roc_score = roc_auc_score(labels, red_ascores)
        roc_res = {'tpr': tpr, 'fpr': fpr, 'ths': thresholds, 'auc': roc_score}
        self.logger.single_plot(
            'roc_curve', tpr, fpr, xlabel='false positive rate', ylabel='true positive rate',
            legend=['auc={}'.format(roc_score)], subdir=subdir
        )
        self.logger.single_save('roc', roc_res, subdir=subdir)
        self.logger.logtxt('##### ROC TEST SCORE {} #####'.format(roc_score), print=True)

        # GTMAPS pixel-wise anomaly detection = explanation performance
        gtmap_roc_res, gtmap_prc_res = None, None
        use_grads = grads is not None
        if gtmaps is not None:
            self.logger.print('Computing GT test score...')
            ascores = self.reduce_pixelwise_ascore(ascores) if not use_grads else grads
            gtmaps = self.test_loader.dataset.dataset.get_original_gtmaps_normal_class()
            if isinstance(self.net, ReceptiveNet):  # Receptive field upsampling for FCDD nets
                ascores = self.net.receptive_upsample(ascores, std=std)
            # Further upsampling for original dataset size
            ascores = torch.nn.functional.interpolate(ascores, (gtmaps.shape[-2:]))
            flat_gtmaps, flat_ascores = gtmaps.reshape(-1).int().tolist(), ascores.reshape(-1).tolist()

            gtfpr, gttpr, gtthresholds = roc_curve(flat_gtmaps, flat_ascores)
            gt_roc_score = roc_auc_score(flat_gtmaps, flat_ascores)
            gtmap_roc_res = {'tpr': gttpr, 'fpr': gtfpr, 'ths': gtthresholds, 'auc': gt_roc_score}
            self.logger.single_plot(
                'gtmap_roc_curve', gttpr, gtfpr, xlabel='false positive rate', ylabel='true positive rate',
                legend=['auc={}'.format(gt_roc_score)], subdir=subdir
            )
            self.logger.single_save(
                'gtmap_roc', gtmap_roc_res, subdir=subdir
            )
            self.logger.logtxt('##### GTMAP ROC TEST SCORE {} #####'.format(gt_roc_score), print=True)

        return {'roc': roc_res, 'gtmap_roc': gtmap_roc_res}

    def heatmap_generation(self, labels: [int], ascores: Tensor, imgs: Tensor,
                           gtmaps: Tensor = None, grads: Tensor = None, show_per_cls: int = 20,
                           name='heatmaps', specific_idx: ([int], [int]) = (), subdir='.'):
        show_per_cls = min(show_per_cls, min(collections.Counter(labels).values()))
        if show_per_cls % 2 != 0:
            show_per_cls -= 1
        lbls = torch.IntTensor(labels)

        # Evaluation Picture with 4 rows. Each row splits into 4 subrows with input-output-heatmap-gtm:
        # (1) 20 first nominal samples (2) 20 first anomalous samples
        # (3) 10 most nominal nominal samples - 10 most anomalous nominal samples
        # (4) 10 most nominal anomalies - 10 most anomalous anomalies
        idx = []
        for l in sorted(set(labels)):
            idx.extend((lbls == l).nonzero().squeeze(-1).tolist()[:show_per_cls])
        rascores = self.reduce_ascore(ascores)
        k = show_per_cls // 2
        for l in sorted(set(labels)):
            lid = set((lbls == l).nonzero().squeeze(-1).tolist())
            sort = [
                i for i in np.argsort(rascores.detach().reshape(rascores.size(0), -1).sum(1)).tolist() if i in lid
            ]
            idx.extend([*sort[:k], *sort[-k:]])
        self._create_heatmaps_picture(
            idx, name, imgs.shape, subdir, show_per_cls, imgs, ascores, grads, gtmaps, labels
        )

        # Concise paper picture: Samples grow from most nominal to most anomalous (equidistant).
        # 2 versions: with local normalization and semi-global normalization
        if 'train' not in name:
            res = self.resdown * 2  # increase resolution limit because there are only a few heatmaps shown here
            rascores = self.reduce_ascore(ascores)
            k = show_per_cls // 3
            inpshp = imgs.shape
            for l in sorted(set(labels)):
                lid = set((torch.from_numpy(np.asarray(labels)) == l).nonzero().squeeze(-1).tolist())
                sort = [
                    i for i in np.argsort(rascores.detach().reshape(rascores.size(0), -1).sum(1)).tolist() if i in lid
                ]
                splits = np.array_split(sort, k)
                idx = [s[int(n / (k - 1) * len(s)) if n != len(splits) - 1 else -1] for n, s in enumerate(splits)]
                self.logger.logtxt(
                    'Interpretation visualization paper image {} indicies for label {}: {}'
                    .format('{}_paper_lbl{}'.format(name, l), l, idx)
                )
                self._create_singlerow_heatmaps_picture(
                    idx, name, inpshp, l, subdir, res, imgs, ascores, grads, gtmaps, labels
                )
                if specific_idx is not None and len(specific_idx) > 0:
                    self._create_singlerow_heatmaps_picture(
                        specific_idx[l], name, inpshp, l, pt.join(subdir, 'specific_viz_ids'),
                        res, imgs, ascores, grads, gtmaps, labels
                    )

    def _create_heatmaps_picture(self, idx: [int], name: str, inpshp: torch.Size, subdir: str,
                                 nrow: int, imgs: Tensor, ascores: Tensor, grads: Tensor, gtmaps: Tensor,
                                 labels: [int], norm: str = 'global'):
        """
        Creates a picture of inputs, heatmaps (either based on ascores or grads, if grads is not None),
        and ground-truth maps (if not None, otherwise omitted). Each row contains nrow many samples.
        One row contains always only one of {input, heatmaps, ground-truth maps}.
        The order of rows thereby is (1) inputs (2) heatmaps (3) ground-truth maps (4) blank.
        For instance, for 20 samples and nrow=10, the picture would show:
            - 10 inputs
            - 10 corresponding heatmaps
            - 10 corresponding ground-truth maps
            - blank
            - 10 inputs
            - 10 corresponding heatmaps
            - 10 corresponding ground-truth maps
        :param idx: limit the inputs (and corresponding other rows) to these indices.
        :param name: name to be used to store the picture.
        :param inpshp: the input shape (heatmaps will be resized to this).
        :param subdir: some subdirectory to store the data in.
        :param nrow: number of images per row.
        :param imgs: the input images.
        :param ascores: anomaly scores.
        :param grads: gradients.
        :param gtmaps: ground-truth maps.
        :param norm: what type of normalization to apply.
            None: no normalization.
            'local': normalizes each heatmap w.r.t. itself only.
            'global': normalizes each heatmap w.r.t. all heatmaps available (without taking idx into account),
                though it is ensured to consider equally many anomalous and nominal samples (if there are e.g. more
                nominal samples, randomly chosen nominal samples are ignored to match the correct amount).
            'semi-global: normalizes each heatmap w.r.t. all heatmaps chosen in idx.
        """
        number_of_rows = int(np.ceil(len(idx) / nrow))
        rows = []
        for s in range(number_of_rows):
            rows.append(self._image_processing(imgs[idx][s * nrow:s * nrow + nrow], inpshp, maxres=self.resdown, qu=1))
            if self.objective != 'hsc':
                err = self.objective != 'ae'
                rows.append(
                    self._image_processing(
                        ascores[idx][s * nrow:s * nrow + nrow], inpshp, maxres=self.resdown, qu=self.quantile,
                        colorize=True, ref=balance_labels(ascores, labels, err) if norm == 'global' else ascores[idx],
                        norm=norm.replace('semi_', ''),  # semi case is handled in the line above
                    )
                )
            if grads is not None:
                rows.append(
                    self._image_processing(
                        grads[idx][s * nrow:s * nrow + nrow], inpshp, self.blur_heatmaps,
                        self.resdown, qu=self.quantile,
                        colorize=True, ref=balance_labels(grads, labels) if norm == 'global' else grads[idx],
                        norm=norm.replace('semi_', ''),  # semi case is handled in the line above
                    )
                )
            if gtmaps is not None:
                rows.append(
                    self._image_processing(
                        gtmaps[idx][s * nrow:s * nrow + nrow], inpshp, maxres=self.resdown, norm=None
                    )
                )
            rows.append(torch.zeros_like(rows[-1]))
        name = '{}_{}'.format(name, norm)
        self.logger.imsave(name, torch.cat(rows), nrow=nrow, scale_mode='none', subdir=subdir)

    def _create_singlerow_heatmaps_picture(self, idx: [int], name: str, inpshp: torch.Size, lbl: int, subdir: str,
                                           res: int, imgs: Tensor, ascores: Tensor, grads: Tensor, gtmaps: Tensor,
                                           labels: [int]):
        """
        Creates a picture of inputs, heatmaps (either based on ascores or grads, if grads is not None),
        and ground-truth maps (if not None, otherwise omitted).
        Row-wise: (1) inputs (2) heatmaps (3) ground-truth maps.
        Creates one version with local normalization and one with semi_global normalization.
        :param idx: limit the inputs (and corresponding other rows) to these indices.
        :param name: name to be used to store the picture.
        :param inpshp: the input shape (heatmaps will be resized to this).
        :param lbl: label of samples (indices), only used for naming.
        :param subdir: some subdirectory to store the data in.
        :param res: maximum allowed resolution in pixels (images are downsampled if they exceed this threshold).
        :param imgs: the input images.
        :param ascores: anomaly scores.
        :param grads: gradients.
        :param gtmaps: ground-truth maps.
        """
        for norm in ['local', 'global']:
            rows = [self._image_processing(imgs[idx], inpshp, maxres=res, qu=1)]
            if self.objective != 'hsc':
                err = self.objective != 'ae'
                rows.append(
                    self._image_processing(
                        ascores[idx], inpshp, maxres=res, colorize=True,
                        ref=balance_labels(ascores, labels, err) if norm == 'global' else None,
                        norm=norm.replace('semi_', ''),  # semi case is handled in the line above
                    )
                )
            if grads is not None:
                rows.append(
                    self._image_processing(
                        grads[idx], inpshp, self.blur_heatmaps, res, colorize=True,
                        ref=balance_labels(grads, labels) if norm == 'global' else None,
                        norm=norm.replace('semi_', ''),  # semi case is handled in the line above
                    )
                )
            if gtmaps is not None:
                rows.append(self._image_processing(gtmaps[idx], inpshp, maxres=res, norm=None))
            tim = torch.cat(rows)
            imname = '{}_paper_{}_lbl{}'.format(name, norm, lbl)
            self.logger.single_save(imname, torch.stack(rows), subdir=pt.join('tims', subdir))
            self.logger.imsave(imname, tim, nrow=len(idx), scale_mode='none', subdir=subdir)

    def _image_processing(self, imgs: Tensor, input_shape: torch.Size, blur: bool = False, maxres: int = 64,
                          qu: float = None, norm: str = 'local', colorize: bool = False, ref: Tensor = None,
                          cmap: str = 'jet', inplace: bool = True) -> Tensor:
        """
        Applies basic image processing techniques, including resizing, blurring, colorizing, and normalizing.
        The resize operation resizes the images automatically to match the input_shape. Other transformations
        are optional. Can be used to create pseudocolored heatmaps!
        :param imgs: a tensor of some images.
        :param input_shape: the shape of the inputs images the data loader returns.
        :param blur: whether to blur the image (has no effect for FCDD anomaly scores, where the
            anomaly scores are upsampled using a Gaussian kernel anyway).
        :param maxres: maximum allowed resolution in pixels (images are downsampled if they exceed this threshold).
        :param norm: what type of normalization to apply.
            None: no normalization.
            'local': normalizes each image w.r.t. itself only.
            'global': normalizes each image w.r.t. to ref (ref defaults to imgs).
        :param qu: quantile used for normalization, qu=1 yields the typical 0-1 normalization.
        :param colorize: whether to colorize grayscaled images using colormaps (-> pseudocolored heatmaps!).
        :param ref: a tensor of images used for global normalization (defaults to imgs).
        :param cmap: the colormap that is used to colorize grayscaled images.
        :param inplace: whether to perform the operations inplace.
        :return: transformed tensor of images
        """
        if not inplace:
            imgs = deepcopy(imgs)
        assert imgs.dim() == len(input_shape) == 4  # n x c x h x w
        std = self.gauss_std
        if qu is None:
            qu = self.quantile

        # upsample if necessary (img.shape != input_shape)
        if imgs.shape[2:] != input_shape[2:]:
            assert isinstance(self.net, ReceptiveNet), \
                'Some images are not of full resolution, and network is not a receptive net. This should not occur! '
            imgs = self.net.receptive_upsample(imgs, reception=True, std=std)

        # blur if requested
        if blur:
            if isinstance(self.net, ReceptiveNet):
                r = self.net.reception['r']
            elif self.objective == 'hsc':
                r = self.net.fcdd_cls(self.net.in_shape, bias=True).reception['r']
            elif self.objective == 'ae':
                enc = self.net.encoder
                if isinstance(enc, ReceptiveNet):
                    r = enc.reception['r']
                else:
                    r = enc.fcdd_cls(enc.in_shape, bias=True).reception['r']
            else:
                raise NotImplementedError()
            r = (r - 1) if r % 2 == 0 else r
            std = std or kernel_size_to_std(r)
            imgs = gaussian_blur2d(imgs, (r,) * 2, (std,) * 2)

        # downsample if resolution exceeds the limit given with maxres
        if maxres < max(imgs.shape[2:]):
            assert imgs.shape[-2] == imgs.shape[-1], 'Image provided is no square!'
            imgs = F.interpolate(imgs, (maxres, maxres), mode='nearest')

        # apply requested normalization
        if norm is not None:
            apply_norm = {
                'local': self.__local_norm, 'global': self.__global_norm
            }
            imgs = apply_norm[norm](imgs, qu, ref)

        # if image is grayscaled, colorize, i.e. provide a pseudocolored heatmap!
        if colorize:
            imgs = imgs.mean(1).unsqueeze(1)
            imgs = colorize_img([imgs, ], norm=False, cmap=cmap)[0]
        else:
            imgs = imgs.repeat(1, 3, 1, 1) if imgs.size(1) == 1 else imgs

        return imgs

    @staticmethod
    def __global_norm(imgs: Tensor, qu: int, ref: Tensor = None) -> Tensor:
        """
        Applies a global normalization of tensor, s.t. the highest value of the complete tensor is 1 and
        the lowest value is >= zero. Uses a non-linear normalization based on quantiles as explained in the appendix
        of the paper.
        :param imgs: images tensor
        :param qu: quantile used
        :param ref: if this is None, normalizes w.r.t. to imgs, otherwise normalizes w.r.t. to ref.
        """
        ref = ref if ref is not None else imgs
        imgs.sub_(ref.min())
        quantile = ref.reshape(-1).kthvalue(int(qu * ref.reshape(-1).size(0)))[0]  # qu% are below that
        imgs.div_(quantile)  # (1 - qu)% values will end up being out of scale ( > 1)
        plosses = imgs.clamp(0, 1)  # clamp those
        return plosses

    @staticmethod
    def __local_norm(imgs: Tensor, qu: int, ref: Tensor = None) -> Tensor:
        """
        Applies a local normalization of tensor, s.t. the highest value of each element (dim=0) in the tensor is 1 and
        the lowest value is >= zero. Uses a non-linear normalization based on quantiles as explained in the appendix
        of the paper.
        :param imgs: images tensor
        :param qu: quantile used
        """
        imgs.sub_(imgs.reshape(imgs.size(0), -1).min(1)[0][(...,) + (None,) * (imgs.dim() - 1)])
        quantile = imgs.reshape(imgs.size(0), -1).kthvalue(
            int(qu * imgs.reshape(imgs.size(0), -1).size(1)), dim=1
        )[0]  # qu% are below that
        imgs.div_(quantile[(...,) + (None,) * (imgs.dim() - 1)])
        imgs = imgs.clamp(0, 1)  # clamp those
        return imgs
