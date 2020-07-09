from fcdd.models.bases import SpatialCenterNet, ReceptiveNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPACEN_AE32(ReceptiveNet):
    def __init__(self, encoder, **kwargs):
        super().__init__(encoder.final_dim, encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder
        self.bn2d0 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, bias=self.bias, padding=1)
        self.bn2d1 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, bias=self.bias, padding=1)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.deconv3 = nn.ConvTranspose2d(32, self.in_shape[0], 3, bias=self.bias, padding=1)

    def forward(self, x):
        x = self.encoder(x, ad=False)
        x = F.leaky_relu(self.bn2d0(x))
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d1(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d2(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


class SPACEN_CNN32_FCONV(SpatialCenterNet):
    pt_cls = SPACEN_AE32

    def __init__(self, final_dim, in_shape, **kwargs):
        super().__init__(final_dim, in_shape **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 32, 5, bias=self.bias, padding=2)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.conv2 = self._create_conv2d(32, 64, 5, bias=self.bias, padding=2)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.conv3 = self._create_conv2d(64, 128, 5, bias=self.bias, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)
        assert 8 * 8 == self.final_dim, "net's final spatial size {}x{} doesn't fit final_dim {}".format(
            8, 8, self.final_dim
        )

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'
        return x

    def get_heatmap(self, pixels, heads=10, threshold=False, invert=False, reception=True, std=None):
        assert pixels.dim() == 4 and pixels.shape[1:] == (1, 8, 8)  # n, heads, h', w'
        return self.conv_final.get_heatmap(pixels, heads, threshold, invert, reception, std=std)  # n, heads, c, h, w


class SPACEN_CNN32_FCONV_S(SpatialCenterNet):
    pt_cls = SPACEN_AE32

    def __init__(self, final_dim, in_shape, **kwargs):
        super().__init__(final_dim, in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 32, 3, bias=self.bias, padding=1)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.conv2 = self._create_conv2d(32, 64, 3, bias=self.bias, padding=1)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.conv3 = self._create_conv2d(64, 128, 3, bias=self.bias, padding=1)
        self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)
        assert 8 * 8 == self.final_dim, "net's final spatial size {}x{} doesn't fit final_dim {}".format(
            8, 8, self.final_dim
        )

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'
        return x

    def get_heatmap(self, pixels, heads=10, threshold=False, invert=False, reception=True, std=None):
        assert pixels.dim() == 4 and pixels.shape[1:] == (1, 8, 8)  # n, heads, h', w'
        return self.conv_final.get_heatmap(pixels, heads, threshold, invert, reception, std=std)  # n, heads, c, h, w
