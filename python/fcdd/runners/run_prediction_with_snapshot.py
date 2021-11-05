import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from fcdd.training.fcdd import FCDDTrainer
from fcdd.models.fcdd_cnn_224 import FCDD_CNN224_VGG_F
from fcdd.datasets.image_folder import ImageFolder
from fcdd.datasets.preprocessing import local_contrast_normalization
from fcdd.util.logging import Logger

# -------- MvTec-AD pre-computed min and max values per class after lcn1 has been applied, copied from fcdd.datasets.mvtec --------
min_max_l1 = [
    [(-1.3336724042892456, -1.3107913732528687, -1.2445921897888184),
     (1.3779616355895996, 1.3779616355895996, 1.3779616355895996)],
    [(-2.2404820919036865, -2.3387579917907715, -2.2896201610565186),
     (4.573435306549072, 4.573435306549072, 4.573435306549072)],
    [(-3.184587001800537, -3.164201259613037, -3.1392977237701416),
     (1.6995097398757935, 1.6011602878570557, 1.5209171772003174)],
    [(-3.0334954261779785, -2.958242416381836, -2.7701096534729004),
     (6.503103256225586, 5.875098705291748, 5.814228057861328)],
    [(-3.100773334503174, -3.100773334503174, -3.100773334503174),
     (4.27892541885376, 4.27892541885376, 4.27892541885376)],
    [(-3.6565306186676025, -3.507692813873291, -2.7635035514831543),
     (18.966819763183594, 21.64590072631836, 26.408710479736328)],
    [(-1.5192601680755615, -2.2068002223968506, -2.3948357105255127),
     (11.564697265625, 10.976534843444824, 10.378695487976074)],
    [(-1.3207964897155762, -1.2889339923858643, -1.148416519165039),
     (6.854909896850586, 6.854909896850586, 6.854909896850586)],
    [(-0.9883341193199158, -0.9822461605072021, -0.9288841485977173),
     (2.290637969970703, 2.4007883071899414, 2.3044068813323975)],
    [(-7.236185073852539, -7.236185073852539, -7.236185073852539),
     (3.3777384757995605, 3.3777384757995605, 3.3777384757995605)],
    [(-3.2036616802215576, -3.221003532409668, -3.305514335632324),
     (7.022546768188477, 6.115569114685059, 6.310940742492676)],
    [(-0.8915618658065796, -0.8669204115867615, -0.8002046346664429),
     (4.4255571365356445, 4.642300128936768, 4.305730819702148)],
    [(-1.9086798429489136, -2.0004451274871826, -1.929288387298584),
     (5.463134765625, 5.463134765625, 5.463134765625)],
    [(-2.9547364711761475, -3.17536997795105, -3.143850803375244),
     (5.305514812469482, 4.535006523132324, 3.3618252277374268)],
    [(-1.2906527519226074, -1.2906527519226074, -1.2906527519226074),
     (2.515115737915039, 2.515115737915039, 2.515115737915039)]
]
# ---------------------------------------------------------------------------------------------------------------------------------


# Path to your snapshot.pt
snapshot = "fcdd/data/mvtec_snapshot.pt"

# Pick the architecture that was used for the snapshot (mvtec's architecture defaults to the following)
net = FCDD_CNN224_VGG_F((3, 224, 224), bias=True).cuda()

# Path to a folder that contains a subfolder containing the images (this is required to use PyTorch's ImageFolder dataset).
# For instance, if the images are in foo/my_images/xxx.png, point to foo. Make sure foo contains only one folder (e.g., my_images).
images_path = "fcdd/data/datasets/foo"

# Pick the class the snapshot was trained on.
normal_class = 0

# Use the same test transform as was used for training the snapshot (e.g., for mvtec, per default, the following)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
    transforms.Normalize(
        min_max_l1[normal_class][0],
        [ma - mi for ma, mi in zip(min_max_l1[normal_class][1], min_max_l1[normal_class][0])]
    )
])

# [optional] to generate heatmaps, define a logger (with the path where the heatmaps should be saved to) and a quantile
logger = None  # Logger("fcdd/data/results/foo")
quantile = 0.97

# Create a trainer to use its loss function for computing anomaly scores
ds = ImageFolder(images_path, transform, transforms.Lambda(lambda x: 0))
loader = DataLoader(ds, batch_size=16, num_workers=0)
trainer = FCDDTrainer(net, None, None, (None, None), logger, 'fcdd', 8, quantile, 224)
trainer.load(snapshot)
trainer.net.eval()
all_anomaly_scores, all_inputs, all_labels = [], [], []
for inputs, labels in loader:
    inputs = inputs.cuda()
    with torch.no_grad():
        outputs = trainer.net(inputs)
        anomaly_scores = trainer.anomaly_score(trainer.loss(outputs, inputs, labels, reduce='none'))
        anomaly_scores = trainer.net.receptive_upsample(anomaly_scores, reception=True, std=8, cpu=False)
        all_anomaly_scores.append(anomaly_scores.cpu())
        all_inputs.append(inputs.cpu())
        all_labels.append(labels)
all_inputs = torch.cat(all_inputs)
all_labels = torch.cat(all_labels)

# all_anomaly_scores will be a tensor containing pixel-wise anomaly scores for all images
all_anomaly_scores = torch.cat(all_anomaly_scores)

# transform the pixel-wise anomaly scores to sample-wise anomaly scores
print(trainer.reduce_ascore(all_anomaly_scores))

# if there is a logger, create heatmaps and save them to the previously defined path using the logger
if logger is not None:
    # show_per_cls defines the maximum number of samples in the heatmaps figures.
    # The heatmap_paper_xxx.png figures, which sort the heatmaps by their anomaly score,
    # use only up to a third of show_per_cls samples.
    trainer.heatmap_generation(all_labels.tolist(), all_anomaly_scores, all_inputs, show_per_cls=1000)
