from fcdd.models.bases import BaseNet
from fcdd.models.fcdd_cnn_32 import FCDD_CNN32_S, FCDD_CNN32_LW3K
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d


class CNN32(BaseNet):
    fcdd_cls = FCDD_CNN32_S

    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)

        self.conv1 = Conv2d(in_shape[0], 32, 5, bias=self.bias, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.pool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(32, 64, 5, bias=self.bias, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.pool2 = MaxPool2d(2, 2)
        self.conv3 = Conv2d(64, 128, 5, bias=self.bias, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool3 = MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512, bias=self.bias)
        self.bn1d1 = nn.BatchNorm1d(512, eps=1e-04, affine=self.bias)
        self.fc_final = nn.Linear(512, 64, bias=self.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool3(F.leaky_relu(self.bn2d3(x)))
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc_final(x)
        return x


class AE32(BaseNet):
    encoder_cls = CNN32

    def __init__(self, encoder, **kwargs):
        super().__init__(encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder
        self.bn1d = nn.BatchNorm1d(64, eps=1e-04, affine=self.bias)
        self.deconv1 = nn.ConvTranspose2d(int(64 / (4 * 4)), 128, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.encoder(x)
        x = self.bn1d(x)
        x = x.view(x.size(0), int(64 / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        return x


class CNN32_LW3K(BaseNet):
    fcdd_cls = FCDD_CNN32_LW3K

    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)

        self.conv1 = Conv2d(in_shape[0], 128, 3, bias=self.bias, padding=1)
        self.bn2d1 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(128, 256, 3, bias=self.bias, padding=1)
        self.bn2d2 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.conv25 = Conv2d(256, 256, 3, bias=self.bias, padding=1)
        self.bn2d25 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.pool2 = MaxPool2d(2, 2)
        self.conv3 = Conv2d(256, 128, 3, bias=self.bias, padding=1)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool3 = MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512, bias=self.bias)
        self.bn1d1 = nn.BatchNorm1d(512, eps=1e-04, affine=self.bias)
        self.fc_final = nn.Linear(512, 64, bias=self.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.conv25(x)
        x = self.pool2(F.leaky_relu(self.bn2d25(x)))
        x = self.conv3(x)
        x = self.pool3(F.leaky_relu(self.bn2d3(x)))
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc_final(x)
        return x


class AE32_LW3K(BaseNet):
    encoder_cls = CNN32_LW3K

    def __init__(self, encoder, **kwargs):
        super().__init__(encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder
        self.bn1d = nn.BatchNorm1d(64, eps=1e-04, affine=self.bias)
        self.deconv1 = nn.ConvTranspose2d(int(64 / (4 * 4)), 128, 3, bias=self.bias, padding=1)
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.deconv2 = nn.ConvTranspose2d(128, 256, 3, bias=self.bias, padding=1)
        self.bn2d5 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.deconv25 = nn.ConvTranspose2d(256, 256, 3, bias=self.bias, padding=1)
        self.bn2d55 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, bias=self.bias, padding=1)
        self.bn2d6 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.deconv4 = nn.ConvTranspose2d(128, 3, 3, bias=self.bias, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bn1d(x)
        x = x.view(x.size(0), int(64 / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.leaky_relu(self.bn2d5(x))
        x = self.deconv25(x)
        x = F.interpolate(F.leaky_relu(self.bn2d55(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        return x
