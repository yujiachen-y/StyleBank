import torch
import torch.nn as nn
import numpy as np


##################
# (1) StyleBankNet
##################
class StyleBankNet(torch.nn.Module):
    def __init__(self, total_style):
        """

        :param total_style: the number of styles need to learn, total_style is 0 meaning is just a encoder and decoder network .
        """
        super(StyleBankNet, self).__init__()

        self.total_style = total_style

        # Non-linearity
        self.relu = nn.ReLU()

        # encoder
        self.enconv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.enconv1_in = InstanceNormalization(32)
        self.enconv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.enconv2_in = InstanceNormalization(64)
        self.enconv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.enconv3_in = InstanceNormalization(128)

        # decoder
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.deconv1_in = InstanceNormalization(64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.deconv2_in = InstanceNormalization(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        # style_bank
        self.style_bank = nn.ModuleList([ConvLayer(128, 128, kernel_size=3, stride=1)
                                         for i in range(total_style)])

    def forward(self, X, style_id=None):
        """

        :param X: input image
        :param style_id: the id of style which network need to transform
        :return: stylized image
        """
        in_X = X
        if style_id is not None:
            if isinstance(style_id, int):
                style_id = [style_id]
                in_X = in_X.unsqueeze(0)

        out = self.relu(self.enconv1_in(self.enconv1(in_X)))
        out = self.relu(self.enconv2_in(self.enconv2(out)))
        out = self.relu(self.enconv3_in(self.enconv3(out)))

        new_out = None
        if style_id is not None:
            # print 'using style mode ... ... ...'
            for i in range(len(style_id)):
                tmp_out = self.style_bank[int(style_id[i]-1)](out[i].unsqueeze(0))
                if new_out is not None:
                    new_out = torch.cat([new_out, tmp_out])
                else:
                    new_out = tmp_out
        else: new_out = out

        out = self.relu(self.deconv1_in(self.deconv1(new_out)))
        out = self.relu(self.deconv2_in(self.deconv2(out)))
        out = self.deconv3(out)

        return out


###################
# (2) Discriminator
###################
NEGATIVE_SLOPE = 0.2
nc, ndf = 3, 64


class Discriminator(nn.Module):
    def __init__(self):

        super(Discriminator, self).__init__()

        # Non-linearity
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Layers
        self.net = nn.Sequential(
            # input is (nc) * 256 * 256
            nn.ReflectionPad2d(1),
            nn.Conv2d(nc, ndf, 4, 2, 0),
            nn.LeakyReLU(NEGATIVE_SLOPE, True),
            nn.BatchNorm2d(ndf),
            # state size. (ndf) * 128 * 128
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 0),
            nn.LeakyReLU(NEGATIVE_SLOPE, True),
            nn.BatchNorm2d(ndf * 2),
            # state size. (ndf*2) * 64 * 64
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 0),
            nn.LeakyReLU(NEGATIVE_SLOPE, True),
            nn.BatchNorm2d(ndf * 4),
            # state size. (ndf*4) * 32 * 32
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 0),
            nn.LeakyReLU(NEGATIVE_SLOPE, True),
            nn.BatchNorm2d(ndf * 8),
            # state size. (ndf*8) * 16 * 16
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 0),
            nn.LeakyReLU(NEGATIVE_SLOPE, True),
            nn.BatchNorm2d(ndf * 8),
            # state size. (ndf*8) * 8 * 8
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 0),
            nn.LeakyReLU(NEGATIVE_SLOPE, True),
            nn.BatchNorm2d(ndf * 8),
            # state size. (ndf * 4) * 4 * 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.net(x)
        return y.view(-1, 1).squeeze(1)


##################
# (3) Other Layers
##################
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2, keepdim=True).unsqueeze(2).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2, keepdim=True).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
