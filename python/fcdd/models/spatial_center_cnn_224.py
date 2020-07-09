from fcdd.models.bases import SpatialCenterNet, ReceptiveNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPACEN_AE224(ReceptiveNet):
    def __init__(self, encoder, **kwargs):
        super().__init__(encoder.final_dim, encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder

        self.bn0 = nn.BatchNorm2d(128, eps=1e-05, affine=self.bias)
        self.deconv1 = nn.ConvTranspose2d(128, 128, 3, bias=self.bias, padding=1)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, bias=self.bias, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, bias=self.bias, padding=1)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.deconv4 = nn.ConvTranspose2d(32, 8, 5, bias=self.bias, padding=2)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.deconv5 = nn.ConvTranspose2d(8, self.in_shape[0], 5, bias=self.bias, padding=2)

    def forward(self, x):
        x  = self.encoder(x, ad=False)
        x = F.leaky_relu(self.bn0(x))
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn1(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x


class SPACEN_CNN224_FCONV(SpatialCenterNet):
    pt_cls = SPACEN_AE224

    def __init__(self, final_dim, in_shape, **kwargs):
        super().__init__(final_dim, in_shape, **kwargs)
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

        assert 28 * 28 == self.final_dim, "net's final spatial size {}x{} doesn't fit final_dim {}".format(
            *self.encoder_out_shape[1:], self.final_dim
        )

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

    def get_heatmap(self, pixels, heads=10, threshold=False, invert=False, reception=True, std=None, cpu=True):
        assert pixels.dim() == 4 and pixels.shape[1:] == (1, *self.encoder_out_shape[1:])  # n, heads, h', w'
        return self.conv_final.get_heatmap(
            pixels, heads, threshold, invert, reception, std=std, cpu=cpu
        )  # n, heads, c, h, w      where c dim contains duplicates
