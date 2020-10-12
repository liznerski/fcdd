from fcdd.models.bases import BaseNet
from fcdd.models.fcdd_cnn_28 import FCDD_CNN28, FCDD_CNN28_W
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d


class CNN28(BaseNet):
    fcdd_cls = FCDD_CNN28

    def __init__(self, in_shape: (int, int, int), **kwargs):
        super().__init__(in_shape, **kwargs)

        self.conv1 = Conv2d(in_shape[0], 8, 5, bias=self.bias, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.pool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(8, 4, 5, bias=self.bias, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=self.bias)
        self.pool2 = MaxPool2d(2, 2)
        self.fc_final = nn.Linear(4 * 7 * 7, 48, bias=self.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2(x)))
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc_final(x)
        return x


class AE28(BaseNet):
    encoder_cls = CNN28

    def __init__(self, encoder: BaseNet, **kwargs):
        super().__init__(encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder
        self.deconv1 = nn.ConvTranspose2d(3, 4, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=self.bias)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=self.bias, padding=3)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 3, 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


class CNN28_W(BaseNet):
    fcdd_cls = FCDD_CNN28_W

    def __init__(self, in_shape: (int, int, int), **kwargs):
        super().__init__(in_shape, **kwargs)

        self.conv1 = Conv2d(in_shape[0], 128, 5, bias=self.bias, padding=2)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(128, 64, 5, bias=self.bias, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.pool2 = MaxPool2d(2, 2)
        self.fc_final = nn.Linear(64 * 7 * 7, 48, bias=self.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2(x)))
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc_final(x)
        return x


class AE28_W(BaseNet):
    encoder_cls = CNN28_W

    def __init__(self, encoder: BaseNet, **kwargs):
        super().__init__(encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder
        self.deconv1 = nn.ConvTranspose2d(3, 64, 5, bias=self.bias, padding=2)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.deconv2 = nn.ConvTranspose2d(64, 128, 5, bias=self.bias, padding=3)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.deconv3 = nn.ConvTranspose2d(128, 1, 5, bias=self.bias, padding=2)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 3, 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x

