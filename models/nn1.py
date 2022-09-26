import torch
import torch.nn as nn
from torch.nn import init
from .MPNCOV.python import MPNCOV
from .layer import CNR2d


class MultiContextModule(nn.Module):
    def __init__(self, in_channels = 32):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="zeros")
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="zeros")
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="zeros")

    def forward(self, x):
        x1 = self.conv1(x)        
        x2 = self.conv2(x1)        
        x3 = self.conv3(x2)
        x_out = torch.cat([x1, x2, x3], dim=1)
        return x_out


class FeatureInterDependenciesModule(nn.Module):
    def __init__(self, channel, reduction=3):
        super(FeatureInterDependenciesModule, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

        self.up_sample = nn.UpsamplingBilinear2d([145, 174])

    def forward(self, x):
        batch_size, c = x.shape[0], x.shape[1]
        x_sub = self.max_pool(x)
        
        cov_mat = MPNCOV.CovpoolLayer(x_sub) # Global Covariance pooling layer
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,c,1,1)
  
        y_cov = self.conv_du(cov_mat_sum)

        x_after_cov = y_cov * x + x
        x_up = self.up_sample(x_after_cov)

        return x_up


class Reconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        return x3



class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mc = MultiContextModule()
        self.fd = FeatureInterDependenciesModule(384)
        self.rec = Reconstruction()

    def forward(self, x):
        x1 = self.mc(x)
        x2 = self.fd(x1)
        x3 = self.rec(x2)
        return x3



class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256
        # dsc4 : 32 x 32 x 256 -> 16 x 16 x 512
        # dsc5 : 16 x 16 x 512 -> 16 x 16 x 1

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        # x = torch.sigmoid(x)

        return x




def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net