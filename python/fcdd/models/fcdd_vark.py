import torch.nn as nn
import torch.nn.functional as F
from fcdd.models.bases import FCDDNet


class FCDD_CNN224_VARK(FCDDNet):
    def __init__(self, in_shape, k=3, **kwargs):
        assert k % 2 == 1, 'kernel size needs to be uneven'
        p = (k - 1) // 2
        super().__init__(in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 32, k, bias=self.bias, padding=p)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(3, 2, 1)  # 32 x 112 x 112

        self.conv2 = self._create_conv2d(32, 128, k, bias=self.bias, padding=p)
        self.bn2d2 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool2 = self._create_maxpool2d(3, 2, 1)  # 128 x 56 x 56

        self.conv3 = self._create_conv2d(128, 256, k, bias=self.bias, padding=p)
        self.bn2d3 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.conv4 = self._create_conv2d(256, 256, k, bias=self.bias, padding=p)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.pool3 = self._create_maxpool2d(3, 2, 1)  # 256 x 28 x 28

        self.conv5 = self._create_conv2d(256, 128, k, bias=self.bias, padding=p)
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


class FCDD_CNN224_3K(FCDD_CNN224_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=3, **kwargs)


class FCDD_CNN224_5K(FCDD_CNN224_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=5, **kwargs)


class FCDD_CNN224_7K(FCDD_CNN224_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=7, **kwargs)


class FCDD_CNN224_9K(FCDD_CNN224_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=9, **kwargs)


class FCDD_CNN224_11K(FCDD_CNN224_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=11, **kwargs)


class FCDD_CNN224_13K(FCDD_CNN224_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=13, **kwargs)


class FCDD_CNN32_VARK(FCDDNet):
    def __init__(self, in_shape, k=3, **kwargs):
        assert k % 2 == 1, 'kernel size needs to be uneven'
        p = (k - 1) // 2
        super().__init__(in_shape, **kwargs)

        self.conv1 = self._create_conv2d(in_shape[0], 128, k, bias=self.bias, padding=p)
        self.bn2d1 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.conv2 = self._create_conv2d(128, 256, 3, bias=self.bias, padding=1)
        self.bn2d2 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.conv3 = self._create_conv2d(256, 128, 3, bias=self.bias, padding=1)
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


class FCDD_CNN32_3K(FCDD_CNN32_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=3, **kwargs)


class FCDD_CNN32_5K(FCDD_CNN32_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=5, **kwargs)


class FCDD_CNN32_7K(FCDD_CNN32_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=7, **kwargs)


class FCDD_CNN32_9K(FCDD_CNN32_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=9, **kwargs)


class FCDD_CNN32_11K(FCDD_CNN32_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=11, **kwargs)


class FCDD_CNN32_13K(FCDD_CNN32_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=13, **kwargs)


class FCDD_CNN32_15K(FCDD_CNN32_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=15, **kwargs)


class FCDD_CNN32_17K(FCDD_CNN32_VARK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, k=17, **kwargs)
