from fcdd.models.bases import ReceptiveNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class AE32(ReceptiveNet):
    def __init__(self, encoder, dropout=None, **kwargs):
        super().__init__(encoder.final_dim, encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder
        self.bn1d = nn.BatchNorm1d(self.encoder.final_dim, eps=1e-04, affine=self.bias)
        self.deconv1 = nn.ConvTranspose2d(int(self.encoder.final_dim / (4 * 4)), 128, 5, bias=self.bias, padding=2)
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
        x = x.view(x.size(0), int(self.encoder.final_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class CNN32(ReceptiveNet):
    pt_cls = AE32

    def __init__(self, final_dim, in_shape, dropout=None, **kwargs):
        super().__init__(final_dim, in_shape, **kwargs)

        self.conv1 = self._create_conv2d(in_shape[0], 32, 5, bias=self.bias, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.conv2 = self._create_conv2d(32, 64, 5, bias=self.bias, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.conv3 = self._create_conv2d(64, 128, 5, bias=self.bias, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.pool3 = self._create_maxpool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512, bias=self.bias)
        self.bn1d1 = nn.BatchNorm1d(512, eps=1e-04, affine=self.bias)
        self.fc_final = nn.Linear(512, self.final_dim, bias=self.bias)

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

    def get_grad_heatmap(self, losses: torch.Tensor, outputs: torch.Tensor, inputs: torch.Tensor,
                         method='grad', absolute=True):
        methods = ('grad', 'xgrad')
        assert method in methods
        grads = torch.autograd.grad((*losses.mean(-1), ), inputs, create_graph=True)[0]
        if method == 'xgrad':
            heatmaps = inputs.detach() * grads
        else:
            heatmaps = grads
        if absolute:
            heatmaps = heatmaps.abs()
        heatmaps = heatmaps.sum(1, keepdim=True)
        # heatmaps /= heatmaps.sum((2, 3), keepdim=True)
        return heatmaps.detach()

