# Explainable Deep One-Class Classification
Here we provide the implementation of *Fully Convolutional Data Description* (FCDD), an explainable approach to deep one-class classification. 
The implementation is based on PyTorch. 

The general idea of FCDD is to make the model a fully convolutional network directly yielding pixel-related anomaly scores. 
Furthermore, we provide an upsampling technique based on the receptive field of the network that outputs full-resolution anomaly heatmaps.
You can find an overview of the method below: 

<img src="data/git_images/fcdd_summary.png?raw=true" height="229" width="430" > 

With the provided code you can use FCDD to produce heatmaps like this:

<img src="data/git_images/fcdd_explanations_mvtec.png?raw=true" height="373" width="633" > 


## Cite the Paper
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

## Installation
It is recommended to use a virtual environment to install FCDD.
Assuming you're in the `python` directory, install FCDD via pip:

    virtualenv -p python3 venv 
    source venv/bin/activate
    pip install .
    
   
## Train FCDD and Reproduce Results

If you have used a virtual environment to install FCDD, make sure to activate it.
We recommend training FCDD by starting one of the runners in `python/fcdd/runners` from the `python/fcdd` directory.
This way the default parameters look for datasets in the `data/datasets` directory and store log data in `data/results`.
If you do not start the runners from `python/fcdd`, you need to give the 
runners the path to the datasets (argument datadir) and the path to the log directory (argument logdir), otherwise
the data might be stored in unexpected locations.

#### Fashion-MNIST
With EMNIST OE:

    python runners/run_fmnist.py --noise-mode emnist
    
With CIFAR-100 OE:
 
    python runners/run_fmnist.py 
    

#### CIFAR-10
For full OE:

    python runners/run_cifar10.py 
    
For limited OE, change the respective parameter, e.g. for using 8 OE samples:

    python runners/run_cifar10.py --oe-limit 8 


#### ImageNet
For full OE:
    
     python runners/run_imagenet.py 
     
Please note that you have to manually download ImageNet1k and ImageNet22k and place them in the correct folders.
Let **dsdir** be your specified dataset directory (per default `../../data/datasets/`). 
ImageNet1k needs to be in `dsdir/imagenet`, containing the devkit, train, and val split in form of a tar file each, with names ILSVRC2012_devkit_t12.tar.gz, ILSVRC2012_img_train.tar, and ILSVRC2012_img_val.tar. 
These are the default names expected by the PyTorch loaders. 
ImageNet22k needs to be in `dsdir/imagenet22k/fall11_whole_extracted`, containing all the extracted class directories with pictures, e.g. the folder n12267677 having pictures of acorns.
You can download ImageNet1k on the official website after having registered: http://image-net.org/download. 
ImageNet22k, i.e. the full release fall 11, can also be found there.

#### MVTec-AD
Using confetti noise:

    python runners/run_mvtec.py 
    
Using a semi-supervised setup with one true anomaly per defection:
    
    python runners/run_mvtec.py --supervise-mode noise --noise-mode mvtec_gt --oe-limit 1 


#### Pascal VOC

    python runners/run_pascalvoc.py 


## Train Baselines

To run the baseline experiments -- i.e. HSC with heatmaps based on gradients and AE with reconstruction error heatmaps -- little has to be done.
Only a few parameters have to be adjusted, most importantly the objective and the network. 
For instance, in the case of Cifar-10 and HSC: 

    python runners/run_cifar10.py --objective hsc -n CNN32  --blur-heatmaps 
    
Similarily for the AE baseline:

    python runners/run_cifar10.py --objective ae -n AE32 --supervise-mode unsupervised --blur-heatmaps 

Alternatively, one can use the `add_exp_to_base.py` script in `python/fcdd/runners`.
It takes a log directory of a previous experiment as input and automatically changes the arguments to 
run baseline experiments with the same basic configuration. 
This script has also the advantage of automatically aggregating heatmaps of the given old experiment and baseline experiments to create combined pictures as seen in the paper.
For instance, let `../../data/results/fcdd_20200801120000_my_awesome_experiment` be a log directory of an old experiment.
Then, to produce both baselines, run:

    python runners/add_exp_to_base.py ../../data/results/fcdd_20200801120000_my_awesome_experiment 

This will create log directories for each baseline by appending a suffix to the given one, i.e. in this case
`../../data/results/fcdd_20200801120000_my_awesome_experiment_HSC` and `../../data/results/fcdd_20200801120000_my_awesome_experiment_AE`.

Note that the previous experiment needs to be an FCDD experiment and not a baseline itself.
Also, note that the baseline experiments use the same data directory parameter as found in the old experiment configuration file. 
That is, if `add_exp_to_base.py` is not started from the same directory (e.g. `python/fcdd`), it will not find the datasets 
and attempt to download it to another one. 

## Explanation of the Log Data / Results

A runner saves the achieved scores, metrics, plots, snapshots, and heatmaps in a given log directory. 
Each log directory contains a subdirectory for each class that is trained to be nominal. 
These subdirectories are named "normal_x", where x is the class number. 
The class subdirectories again contain a subdirectory for each random seed.
These are named "it_x" for x being the iteration number (random seed). 
Inside the seed subdirectories all actual log data can be found. 
Additionally, summarized plots will be created for the class subdirectories and the root log directory. 
For instance, a plot containing ROC curves for each class (averaged over all seeds) can be found in the root log directory. 

Visualization for 2 classes and 2 random seeds: \
./log_directory \
./log_directory/normal_0 \
./log_directory/normal_0/it_0 \
./log_directory/normal_0/it_1 \
./log_directory/normal_1 \
./log_directory/normal_1/it_0 \
./log_directory/normal_1/it_1 \
...

The actual log data consists of: 
- config.txt: A file containing all the arguments of the training.
- ds_preview.png: A preview of the dataset, i.e. some random nominal and anomalous samples from the training set. Includes augmentation and data preprocessing. Also shows corresponding ground-truth maps, if such are available. 
- err.pdf: A plot of the loss over time.
- err_anomalous.pdf: A plot of the loss for anomalous samples only.
- err_normal.pdf: A plot of the loss for nominal samples only.
- heatmaps_paper_x_lbly.png: An image of test heatmaps, test inputs, and ground-truth maps. One image for each x-y combination. x is the normalization used, either local (each heatmap is normalized w.r.t. itself only) or semi_global (each heatmap is normalized w.r.t. to all heatmaps in the image). y is the label, i.e. either 1 or 0 (per default 1 is for anomalies).
- heatmaps_semi_global.png: An image of the first test heatmaps and inputs found in the dataset. The first row shows nominal samples, the second-row anomalous samples. The third row shows the ten most nominal rated nominal samples on the left and the ten most anomalous rated nominal samples on the right. The fourth row shows the ten most nominal rated anomalies on the left and the ten most anomalous rated anomalies on the right. Note that changing the nominal label to 1 flips this. 
- train_heatmaps_semi_global.png: Like above, but for training samples.
- history.json: A file containing metrics in text form.
- log.txt: A file containing some logged txt lines, like the average AUC value achieved and the duration of the training.
- print.log: Logs all that has been printed using the Logger.
- roc.json: ROC values saved as text (not very readable for humans). 
- roc_curve.pdf: ROC for detection performance.
- gtmap_roc.pdf: ROC for explanation performance. Only for MVTec-AD (or datasets with ground-truth maps). 
- snapshot.pt: Snapshot of the training state and model parameters.
- src.tar.gz: Snapshot of the complete source code at the moment of the training start time. 
- tims: A directory containing raw tensors heatmaps (not readable for humans). 

## Customize FCDD 

In general, the FCDD implementation splits in five packages: `datasets`, `models`, `runners`, `training`, and `util`. 
`dataset` contains an implementation of the base class for AD datasets, actual torchvision-style dataset implementations, and artificial anomaly generation implementations. 
`models` contains an implementation of the network base class -- including receptive field upsampling and gradient-based heatmap computation -- and implementations of all network architectures.
`runners` contains scripts for starting training runs, processing program arguments, and preparing all training parameters -- like creating an optimizer and network instance.
`training` contains an implementation for actual training and evaluation of the network.
`util` contains all the rest, i.e. e.g. a logger that handles all I/O interactions. 

In the following we give a brief tutorial on how to modify the FCDD implementation for specific requirements.

#### Add a new Network Architecture
1) Create a new python script in the `models` package. 
Implement a PyTorch module that inherits the `fcdd.models.bases.BaseNet` or the `fcdd.models.bases.ReceptiveNet` class. 
The latter takes track of the receptive field. Read the documentation for further details. 

2) Import the new network class in `fcdd.models.__init__` to automatically add the network name to the available ones. 
Make sure that the class name is unique. 

#### Add a new Dataset
1) Create a new python script in the `datasets` package. 
Implement a dataset that inherits the `fcdd.datasets.bases.TorchvisionDataset` class. 
Your implementation needs to process all parameters of the `fcdd.datasets.bases.load_dataset` function in its initialization. 
You can use the *preproc* parameter to switch between different data preprocessing pipelines (augmentation, etc). 
In the end, your implementation needs to have at least all attributes defined in `fcdd.datasets.bases.BaseADDataset` class. 
Most importantly, the *_train_set* attribute and the *_test_set* attribute containing the corresponding torchvision-style datasets. 
Have a look at the already available implementations. 

2) Add a name for your dataset to the `fcdd.datasets.__init__.DS_CHOICES` variable. 
Add your dataset to the "switch-case" in the `fcdd.datasets.__init__.load_dataset` function. 
Add the number of available class for your dataset to the `fcdd.datasets.__init__.no_classes` function and 
add the class names to `fcdd.datasets.__init__.str_labels`. 

#### Add a new Training Parameter
1) Add an argument to the `fcdd.runners.argparse_configs.DefaultConfig` class. 

2) Add this argument as a parameter to the `fcdd.training.setup.trainer_setup` function. 
Also, make it return the parameter itself, or some other thing that has been created based on your new argument (e.g. an instance of some network, optimizer, etc.).

3) Add whatever new is being returned by the `fcdd.training.setup.trainer_setup` function as a parameter to the `fcdd.training.super_trainer.SuperTrainer` class.

4) Implement your requested novel behavior based on the new parameter in the SuperTrainer or one of the actual model trainers (FCDDTrainer, HSCTrainer, AETrainer), depending on the objective.

#### Add a new Runner

1) Create a new python script in the `runners` package. 
Implement a configuration that inherits `fcdd.runners.argparse_configs.DefaultConfig`. 
You need to implement the __call__ method for it, where you can add new arguments to the given parser or change the default values. 

2) Create an instance of one of the runners, e.g. `fcdd.runners.bases.ClassesRunner` or  `fcdd.runners.bases.SeedsRunner`, and use your configuration implementation for that. 
For both, the SeedsRunner and ClassesRunner, you need to additionally add an argument in your configuration named **it** that determines the number of different random seeds.
Invoke the instance's run method. 


#### Add a new Objective / Trainer

1) Add a string identification for your new objective to the `fcdd.training.setup.OBJECTIVES` variable. 

2) Create a new trainer script in the `training` package. 
Implement a trainer inheriting `fcdd.training.bases.BaseADTrainer`. 
At least implement the loss and snapshot method. 

3) Add the creation of an instance for your new trainer to the "switch-case" in the initialization of `fcdd.training.super_trainer.SuperTrainer`. 

#### Add a new Type of Synthetic Anomaly
1) Implement your new synthetic anomaly in `fcdd.datasets.noise`. 

2) Add a string identification for you new function to the `fcdd.datasets.noise_modes.MODES` variable.

3) Add an invocation of your new function to the "switch-case" in `fcdd.datasets.noise_modes.generate_noise`. 

#### Add a new Outlier Exposure Dataset

1) Create a new python script in the `datasets/outlier_exposure` package. 
Implement a torchvision-style dataset that expects at least the following parameters: **size**, **root**, and **limit_var**.
Also, you need to at least implement a __get_item__ method and *data_loader* method. 
Cf. one of the existing implementations of Outlier Exposure datasets, e.g. `fcdd.datasets.outlier_exposure.emnist`.

## Need help?
If you find any bugs, have questions, or need help modifying FCDD, feel free to write us an [email](mailto:p_liznersk13@cs.uni-kl.de)!

