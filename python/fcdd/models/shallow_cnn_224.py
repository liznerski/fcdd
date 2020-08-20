from fcdd.models.bases import BaseNet
from fcdd.models.fcdd_cnn_224 import FCDD_CNN224
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d


class CNN224(BaseNet):
    fcdd_cls = FCDD_CNN224

    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.conv1 = Conv2d(in_shape[0], 8, 5, bias=self.bias, padding=2)
        self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.pool1 = MaxPool2d(3, 2, 1)  # 8 x 112 x 112

        self.conv2 = Conv2d(8, 32, 5, bias=self.bias, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.pool2 = MaxPool2d(3, 2, 1)  # 32 x 56 x 56

        self.conv3 = Conv2d(32, 64, 3, bias=self.bias, padding=1)
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.conv4 = Conv2d(64, 128, 3, bias=self.bias, padding=1)
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool3 = MaxPool2d(3, 2, 1)  # 128 x 28 x 28

        self.conv5 = Conv2d(128, 128, 3, bias=self.bias, padding=1)
        self.bn2d5 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool4 = MaxPool2d(3, 2, 1)  # 128 x 14 x 14

        self.conv6 = Conv2d(128, 64, 3, bias=self.bias, padding=1)
        self.bn2d6 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.pool5 = MaxPool2d(3, 2, 1)  # 64 x 7 x 7

        self.fc1 = nn.Linear(64 * 7 * 7, 1536, bias=self.bias)
        self.bn1d1 = nn.BatchNorm1d(1536, eps=1e-04, affine=self.bias)
        self.fc_final = nn.Linear(1536, 784, bias=self.bias)

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
        x = F.leaky_relu(self.bn2d5(x))
        x = self.pool4(x)

        x = self.conv6(x)
        x = F.leaky_relu(self.bn2d6(x))
        x = self.pool5(x)

        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc_final(x)
        return x


class AE224(BaseNet):
    encoder_cls = CNN224

    def __init__(self, encoder, **kwargs):
        super().__init__(encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder

        self.bn1d = nn.BatchNorm1d(784, eps=1e-04, affine=self.bias)
        self.deconv1 = nn.ConvTranspose2d(int(784 / (7 * 7)), 64, 3, bias=self.bias, padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2dd1 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)

        self.deconv2 = nn.ConvTranspose2d(64, 128, 3, bias=self.bias, padding=1)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2dd2 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)

        self.deconv3 = nn.ConvTranspose2d(128, 128, 3, bias=self.bias, padding=1)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2dd3 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)

        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, bias=self.bias, padding=1)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2dd4 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, bias=self.bias, padding=1)
        nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2dd5 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)

        self.deconv6 = nn.ConvTranspose2d(32, 8, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2dd6 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)

        self.deconv7 = nn.ConvTranspose2d(8, self.encoder.in_shape[0], 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv7.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x, ad=True):
        x = self.encoder(x)
        x = self.bn1d(x)
        x = x.view(x.size(0), int(784 / (7 * 7)), 7, 7)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2dd1(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2dd2(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2dd3(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.leaky_relu(self.bn2dd4(x))
        x = self.deconv5(x)
        x = F.interpolate(F.leaky_relu(self.bn2dd5(x)), scale_factor=2)
        x = self.deconv6(x)
        x = F.interpolate(F.leaky_relu(self.bn2dd6(x)), scale_factor=2)
        x = self.deconv7(x)
        x = torch.sigmoid(x)
        return x
