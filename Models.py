import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import ssl
import numpy as np
import math
ssl._create_default_https_context = ssl._create_unverified_context
__all__ = [
    "ResidualConvBlock",
    "Discriminator", "Generator",
    "ContentLoss"
]

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(1, 1, kernel_size, stride=stride, padding=padding, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)

class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=True):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        img = img.type(torch.float32)
        return img


class New_D_doubleconv(nn.Module):
    def __init__(self,dropout = None) :
        self.dropout_val = dropout
        super(New_D_doubleconv, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1,16, 4,padding='same')
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 4,padding='same')
        self.conv3 = nn.Conv2d(16, 64, 4,padding='same')
        self.conv4 = nn.Conv2d(64, 64, 4,padding='same')
        self.conv5 = nn.Conv2d(64, 128, 4,padding='same')
        self.conv6 = nn.Conv2d(128, 128, 4,padding='same')
        self.conv7 = nn.Conv2d( 128, 256, 4,padding='same')
        self.conv8 = nn.Conv2d(256, 256, 4,padding='same')
        self.conv9 = nn.Conv2d( 128, 256, 4,padding='same')
        self.conv10 = nn.Conv2d(256, 256, 4,padding='same')
        self.drop_const_1 = nn.Dropout(0.3)
        self.drop_const_2 = nn.Dropout(0.5)
        #self.dropout = nn.Dropout(self.dropout_val)
        self.fc1 = nn.Linear(256 * 4 *4, 512)
        self.fc1_ = nn.Linear(256 * 3 * 3, 512)
        self.fc1__ = nn.Linear(256 * 6 *6, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(512, 1)

        # self.fc1 = nn.Linear(256 * 4 *4, 512)
        # self.fc1_ = nn.LazyLinear( 512)
        # self.fc2 = nn.LazyLinear(64)
        # self.fc3 = nn.LazyLinear( 1)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        # add sequence of convolutional and max pooling layers
        #input size [batch,1,240,240]
        x_temp = F.relu(self.conv1(input))
        # current state [batch,16,240,240]
        x = self.pool(F.relu(self.conv2(x_temp))+x_temp)
        # current state [batch,16,120,120]
        x_temp = F.relu(self.conv3(x))
        # current state [batch,32,120,120]
        x = self.pool(F.relu(self.conv4(x_temp))+x_temp)
        # current state [batch,32,60,60]
        # if self.dropout_val is not None:
        #     x = self.dropout(x)
        x_temp = F.relu(self.conv5(x))
        # current state [batch,64,60,60]
        x = self.pool(F.relu(self.conv6(x_temp))+x_temp)
        # current state [batch,64,30,30]
        x_temp = F.relu(self.conv7(x))
        # # current state [batch,128,30,30]
        x = self.pool(F.relu(self.conv8(x_temp))+x_temp)
        x = self.drop_const_1(x)


        # if self.dropout_val is not None:
        #     x = self.dropout(x)
        # current state [batch,256,7,7]
        if input.size()[-1] == 64:
            x = x.view(-1, 256 * 4 *4)
            x = F.leaky_relu(self.fc1(x))
        elif input.size()[-1] == 48:
            x = x.view(-1, 256 * 3 * 3)
            x = F.leaky_relu(self.fc1_(x))
        elif input.size()[-1] == 96:
            x = x.view(-1, 256 * 6* 6)
            x = F.leaky_relu(self.fc1__(x))
        #if self.dropout_val is not None:
        x = self.drop_const_2(x)
        #x = F.leaky_relu(self.fc2(x))

        x = self.fc3(x)
        return x

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):

        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp



