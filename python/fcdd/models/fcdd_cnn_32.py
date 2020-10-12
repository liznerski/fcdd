from fcdd.models.bases import FCDDNet, BaseNet
import torch.nn as nn
import torch.nn.functional as F


class FCDD_CNN32(FCDDNet):
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 32, 5, bias=self.bias, padding=2)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.conv2 = self._create_conv2d(32, 64, 5, bias=self.bias, padding=2)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.conv3 = self._create_conv2d(64, 128, 5, bias=self.bias, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'
        return x


class FCDD_CNN32_LW3K(FCDDNet):
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 128, 3, bias=self.bias, padding=1)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.bn2d1 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.conv2 = self._create_conv2d(128, 256, 3, bias=self.bias, padding=1)
        self.bn2d2 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.conv25 = self._create_conv2d(256, 256, 3, bias=self.bias, padding=1)
        self.bn2d25 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.conv3 = self._create_conv2d(256, 128, 3, bias=self.bias, padding=1)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.conv25(x)
        x = self.pool2(F.leaky_relu(self.bn2d25(x)))
        x = self.conv3(x)
        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'
        return x


class FCDD_CNN32_S(FCDDNet):
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 32, 3, bias=self.bias, padding=1)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.conv2 = self._create_conv2d(32, 64, 3, bias=self.bias, padding=1)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.conv3 = self._create_conv2d(64, 128, 3, bias=self.bias, padding=1)
        self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'
        return x
