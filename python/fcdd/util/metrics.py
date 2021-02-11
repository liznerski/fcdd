from typing import List

import numpy as np


def mean_roc(results: List[dict]):
    """
    This computes a "mean" of multiple ROCs. While the mean of the AuROC is precise,
    the mean curve is rather an approximation.
    :param results:
        Result dictionary of the following form: {
            'tpr': [], 'fpr': [], 'ths': [], 'auc': int, ...
        }
    :return: mean of all ROCs in form of another result dictionary
    """
    if results is None or any([r is None for r in results]) or len(results) == 0:
        return None
    results = results.copy()
    tpr, fpr, ths, auc = [], [], [], []
    x = 'fpr'
    y = 'tpr'
    for res in results:
        try:
            tpr.append(res[y].tolist())
            fpr.append(res[x].tolist())
            ths.append(res['ths'].tolist())
            auc.append(res['auc'].tolist())
        except AttributeError:
            tpr.append(res[y])
            fpr.append(res[x])
            ths.append(res['ths'])
            auc.append(res['auc'])

    ml = min([len(arr) for arr in ths])
    for i in range(len(ths)):
        d = len(ths[i]) - ml
        pick = np.random.binomial(1, (len(ths[i]) - d) / len(ths[i]), size=len(ths[i])).nonzero()
        tpr[i] = (np.asarray(tpr[i])[pick]).tolist()
        fpr[i] = (np.asarray(fpr[i])[pick]).tolist()
        ths[i] = (np.asarray(ths[i])[pick]).tolist()

    ml = min([len(arr) for arr in ths])
    for n in range(len(ths)):
        while len(ths[n]) > ml:
            rmv = np.random.randint(len(ths[n]))
            del tpr[n][rmv], fpr[n][rmv], ths[n][rmv]
        tpr[n], fpr[n], ths[n] = np.asarray(tpr[n]), np.asarray(fpr[n]), np.asarray(ths[n])

    tpr, fpr, ths = np.asarray(tpr), np.asarray(fpr), np.asarray(ths)
    tpr, fpr, ths = np.mean(tpr, axis=0), np.mean(fpr, axis=0), np.mean(ths, axis=0)
    auc = np.mean(auc)
    return {
        y: tpr, x: fpr, 'ths': ths, 'auc': auc
    }
