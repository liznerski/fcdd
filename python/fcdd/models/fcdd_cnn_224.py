import os.path as pt

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fcdd.models.bases import FCDDNet, BaseNet
from torch.hub import load_state_dict_from_url


class FCDD_CNN224_VGG_NOPT(FCDDNet):
    # VGG_11BN based net with randomly initialized weights (pytorch default).
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, 'VGG net is only supported with bias atm!'

        self.features = nn.Sequential(
            self._create_conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            self._create_conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # CUT
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.features = self.features[:-8]

        self.conv_final = self._create_conv2d(512, 1, 1)

    def forward(self, x, ad=True):
        x = self.features(x)

        if ad:
            x = self.conv_final(x)

        return x


class FCDD_CNN224_VGG(FCDDNet):
    # VGG_11BN based net with most of the VGG layers having weights pretrained on the ImageNet classification task.
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, 'VGG net is only supported with bias atm!'
        state_dict = load_state_dict_from_url(
            torchvision.models.vgg.model_urls['vgg11_bn'],
            model_dir=pt.join(pt.dirname(__file__), '..', '..', '..', 'data', 'models')
        )
        features_state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith('features')}

        self.features = nn.Sequential(
            self._create_conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            self._create_conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            self._create_maxpool2d(2, 2),
            # Frozen version freezes up to here
            self._create_conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            self._create_conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # CUT
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.features.load_state_dict(features_state_dict)
        self.features = self.features[:-8]

        self.conv_final = self._create_conv2d(512, 1, 1)

    def forward(self, x, ad=True):
        x = self.features(x)

        if ad:
            x = self.conv_final(x)

        return x


class FCDD_CNN224_VGG_F(FCDD_CNN224_VGG):
    # VGG_11BN based net with most of the VGG layers having weights pretrained on the ImageNet classification task.
    # Additionally, these weights get frozen, i.e., the weights will not get updated during training.
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        for m in self.features[:15]:
            for p in m.parameters():
                p.requires_grad = False


class FCDD_CNN224(FCDDNet):
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 8, 5, bias=self.bias, padding=2)
        self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(3, 2, 1)  # 8 x 112 x 112

        self.conv2 = self._create_conv2d(8, 32, 5, bias=self.bias, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.pool2 = self._create_maxpool2d(3, 2, 1)  # 32 x 56 x 56

        self.conv3 = self._create_conv2d(32, 64, 3, bias=self.bias, padding=1)
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.conv4 = self._create_conv2d(64, 128, 3, bias=self.bias, padding=1)
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool3 = self._create_maxpool2d(3, 2, 1)  # 128 x 28 x 28

        self.conv5 = self._create_conv2d(128, 128, 3, bias=self.bias, padding=1)
        self.encoder_out_shape = (128, 28, 28)
        self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2d1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.leaky_relu(self.bn2d3(x))
        x = self.conv4(x)
        x = F.leaky_relu(self.bn2d4(x))
        x = self.pool3(x)

        x = self.conv5(x)

        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'

        return x


class FCDD_CNN224_W(FCDDNet):
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 32, 5, bias=self.bias, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(3, 2, 1)  # 32 x 112 x 112

        self.conv2 = self._create_conv2d(32, 128, 5, bias=self.bias, padding=2)
        self.bn2d2 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool2 = self._create_maxpool2d(3, 2, 1)  # 128 x 56 x 56

        self.conv3 = self._create_conv2d(128, 256, 3, bias=self.bias, padding=1)
        self.bn2d3 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.conv4 = self._create_conv2d(256, 256, 3, bias=self.bias, padding=1)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.pool3 = self._create_maxpool2d(3, 2, 1)  # 256 x 28 x 28

        self.conv5 = self._create_conv2d(256, 128, 3, bias=self.bias, padding=1)
        self.encoder_out_shape = (128, 28, 28)
        self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2d1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.leaky_relu(self.bn2d3(x))
        x = self.conv4(x)
        x = F.leaky_relu(self.bn2d4(x))
        x = self.pool3(x)

        x = self.conv5(x)

        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'

        return x
