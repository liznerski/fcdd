from fcdd.models.bases import ReceptiveNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class AE28(ReceptiveNet):
    def __init__(self, encoder, dropout=None, **kwargs):
        super().__init__(encoder.final_dim, encoder.in_shape, bias=encoder.bias, **kwargs)
        self.encoder = encoder
        self.deconv1 = nn.ConvTranspose2d(int(self.encoder.final_dim / (4 * 4)), 4, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=self.bias)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=self.bias, padding=3)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), int(self.encoder.final_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


class CNN28(ReceptiveNet):
    pt_cls = AE28
    """
    Simple CNN. Consists out of a specified number of  Multihead Attention Layers and 2 conv layers.
    Works for 28x28 images.

    :param dim: final output dimension, i.e. embedding size
    :param chin: input shape c x h x w
    :param att_pos: iterable of position, where to insert attentions at, each must be in [0, 4] (amount of layers)
        where 0 means before fist conv and 5 after last one, negative means start counting from the end of the net
    """
    def __init__(self, final_dim, in_shape, dropout=None, **kwargs):
        super().__init__(final_dim, in_shape, **kwargs)

        # general cnn architecture
        self.conv1 = self._create_conv2d(in_shape[0], 8, 5, bias=self.bias, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
        self.pool1 = self._create_maxpool2d(2, 2)
        self.conv2 = self._create_conv2d(8, 4, 5, bias=self.bias, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=self.bias)
        self.pool2 = self._create_maxpool2d(2, 2)
        self.fc_final = nn.Linear(4 * 7 * 7, final_dim, bias=self.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2(x)))
        x = x.contiguous().view(x.size(0), -1)
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
