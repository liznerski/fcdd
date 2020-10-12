import torch.nn as nn
import torch.nn.functional as F
from fcdd.models.bases import FCDDNet


class FCDD_CNN28(FCDDNet):
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 8, 5, bias=self.bias, padding=2)
        self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.conv2 = self._create_conv2d(8, 16, 5, bias=self.bias, padding=2)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.conv_final = self._create_conv2d(16, 1, 1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool2(x)
        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'
        return x


class FCDD_CNN28_W(FCDDNet):
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 128, 5, bias=self.bias, padding=2)
        self.bn2d1 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.conv2 = self._create_conv2d(128, 128, 5, bias=self.bias, padding=2)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool2(x)
        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'
        return x
