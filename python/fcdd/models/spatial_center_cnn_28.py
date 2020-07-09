from fcdd.models.bases import SpatialCenterNet, ReceptiveNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPACEN_AE28(ReceptiveNet):
    def __init__(self, encoder, **kwargs):
        super().__init__(encoder.final_dim, encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder
        self.deconv1 = nn.ConvTranspose2d(16, 8, 5, bias=self.bias, padding=2)
        self.bn3 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.deconv2 = nn.ConvTranspose2d(8, self.in_shape[0], 5, bias=self.bias, padding=2)

    def forward(self, x):
        x = self.encoder(x, ad=False)
        x = F.leaky_relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = torch.sigmoid(x)
        return x


class SPACEN_CNN28_FCONV(SpatialCenterNet):
    pt_cls = SPACEN_AE28

    def __init__(self, final_dim, in_shape, **kwargs):
        super().__init__(final_dim, in_shape, **kwargs)
        self.conv1 = self._create_conv2d(in_shape[0], 8, 5, bias=self.bias, padding=2)
        self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.conv2 = self._create_conv2d(8, 16, 5, bias=self.bias, padding=2)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.conv_final = self._create_conv2d(16, 1, 1, bias=self.bias)
        assert 7 * 7 == self.final_dim, "net's final spatial size {}x{} doesn't fit final_dim {}".format(
            7, 7, self.final_dim
        )

    def forward(self, x, ad=True):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool2(x)
        if ad:
            x = self.conv_final(x)  # n x heads x h' x w'
        return x

    def get_heatmap(self, pixels, heads=10, threshold=False, invert=False, reception=True, std=None):
        assert pixels.dim() == 4 and pixels.shape[1:] == (1, 7, 7)  # n, heads, h', w'
        return self.conv_final.get_heatmap(pixels, heads, threshold, invert, reception, std)  # n, heads, c, h, w
        # where c dim contains duplicates

