# Explainable Deep One-Class Classification
Here we provide the implementation of *Fully Convolutional Data Description* (FCDD), an explainable approach to deep one-class classification. The implementation is based on PyTorch. 

## Citation
A preprint of our paper is available at: https://arxiv.org/abs/2007.01760 

If you use our work, please also cite the current preprint:
```
@misc{liznerski2020explainable,
    title={Explainable Deep One-Class Classification},
    author={Philipp Liznerski and Lukas Ruff and Robert A. Vandermeulen and Billy Joe Franks and Marius Kloft and Klaus-Robert Müller},
    year={2020},
    eprint={2007.01760},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Version
Please note that the current code is a preliminary version! \
For instance, the parameters for running a training are suboptimally structured and poorly documented. \
We will provide an updated version with thorough documentation and cleaned up code in the near future. \
Stay tuned!

## Installation
We recommended using a virtual environment to install FCDD.
Assuming you're in the `python` directory, install FCDD via pip:

    virtualenv -p python3 venv 
    source venv/bin/activate
    pip install .
    
## Train FCDD

In general, any of the runners in `python/fcdd/runners/bases.py` can be used. \
The BaseRunner performs just one training for one class and seed. \
The SeedRunner repeats x training runs for x seeds. \
The ClassRunner trains all classes with x seeds each. \
But the easiest way is to use one of the prepared dataset runners in `python/fcdd/runners`, where each dataset runner sets some default parameters to train specific datasets. 

A runner downloads the required datasets (but imagenet, which is no longer publicly available) automatically and stores them in `../../data/datasets`.
It logs all relevant results (errors, AUCs, heatmap pictures, snapshots, ...) in `../../data/results/experiment_folder_with_uid`.
The location of those both folders can be changed by altering the parameters **datadir** and **logdir** of the runner.
For instance, run:

    python runners/run_fmnist.py --datadir my_awesome_data_folder --logdir my_awesome_log_folder
    
The default parameters assume the code is run from the `python/fcdd` directory and store data in `data`. 
Thus, please make sure to either be in the `python/fcdd` directory or change the **datadir** and **logdir** parameters accordingly to avoid large data being stored in unexpected locations.
In the following we assume you run code from the `python/fcdd` directory.
   
## Reproduce Results

#### Fashion-MNIST
With EMNIST OE:

    python runners/run_fmnist.py -e 400 --supervise-mode noise --supervise-params noise_mode=emnist online=1 --bias --preproc aeaug1
    
With CIFAR-100 OE:
 
    python runners/run_fmnist.py -e 400 --supervise-mode noise --supervise-params noise_mode=cifar100 online=1 --bias --preproc aeaug1
    

#### CIFAR-10
For full OE:

    python runners/run_cifar10.py -e 600 -b 20 --acc-batches 10 --lr-sched-param 0.1 400 500 --optimizer-type adam --scheduler-type milestones --supervise-mode noise --supervise-params noise_mode=cifar100 online=1 --bias --preproc aug1

For limited OE, change respective parameter in supervise-params, e.g. for using 8 OE samples:

    python runners/run_cifar10.py -e 600 -b 20 --acc-batches 10 --lr-sched-param 0.1 400 500 --optimizer-type adam --scheduler-type milestones --supervise-mode noise --supervise-params noise_mode=cifar100 online=1 limit=8 --bias --preproc aug1


#### ImageNet
For full OE:
    
     python runners/run_imagenet.py -e 600 -b 20 --acc-batches 10 --lr-sched-param 0.1 400 500 --optimizer-type adam --scheduler-type milestones --supervise-mode noise --supervise-params noise_mode=imagenet22k online=1 --bias --preproc aug1

Please note that you have to manually download ImageNet1k and ImageNet22k and place them in the correct folders.
Assuming your dataset folder is `data/datasets`. 
ImageNet1k needs to be in `data/datasets/imagenet`, containing the devkit, train, and val split in the form of a tar file each, with names ILSVRC2012_devkit_t12.tar.gz, ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar. 
These are the default names expected by the PyTorch loaders. 
ImageNet22k needs to be in `data/datasets/imagenet22k/fall11_whole_extracted`, containing all the extracted class directories with pictures, e.g. the folder n12267677 having pictures of acorns.
You can download ImageNet1k on the official website after having registered: http://image-net.org/download. 
ImageNet22k, i.e. the full release fall 11, can also be found there.

#### MVTec-AD
Using confetti noise:

    python runners/run_mvtec.py -e 200 -b 16 --acc-batches 8 --supervise-mode malformed_normal --supervise-params noise_mode=confetti online=1 --objective-params gaus_std=12 -wd 1e-5 --bias --preproc aeaug1
    
Using a semi-supervised setup with one true anomaly per defection:
    
    python runners/run_mvtec.py -e 200 -b 16 --acc-batches 8 --supervise-mode noise --supervise-params noise_mode=mvtec_gt online=1 limit=1 -wd 1e-5 --bias --preproc aeaug1


#### Pascal VOC

    python runners/run_pascalvoc.py -e 600 -b 20 --acc-batches 10 --lr-sched-param 0.1 400 500 --optimizer-type adam --scheduler-type milestones --supervise-mode noise --supervise-params noise_mode=imagenet_for_voc online=1 nominal_label=1 --bias --preproc aug1


