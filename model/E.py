import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from model.resblocks import Block
from model.resblocks import OptimizedBlock


class E(nn.Module):

    def __init__(self, num_features=64, activation=F.relu):
        super(E, self).__init__()
        self.num_features = num_features
        self.activation = activation

        self.block1 = OptimizedBlock(6, num_features) # 128
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)  # 64
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True) # 32
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True) # 16
        self.block5 = Block(num_features * 8, num_features * 8,
                            activation=activation, downsample=True)  # 8
        self.l6 = utils.spectral_norm(nn.Linear(num_features * 8, 256))
        self.l7 = utils.spectral_norm(nn.Linear(num_features * 8, 256))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x,):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        h = self.activation(h)   # 512
        e = torch.mean(h,dim=0,keepdim=True)
        out1 = self.l6(h)  # B,256  P矩阵
        size = out1.size()  # B,256
        assert len(size)==2, 'size is not right'
        mean = out1.view(size[0],size[1],1,1)  # 512维
        mean = torch.mean(mean,dim=0,keepdim=True)
        std = self.l7(h) # B,256
        std = std.view(size[0],size[1],1,1)
        std = torch.mean(std,dim=0,keepdim=True)
        return e, mean, std


if __name__=='__main__':
    model = E().cuda()
    data = torch.randn(8,6,256,256).cuda()
    output = model(data)
    e = output[0]
    mean = output[1]
    print(e.size())  # torch.Size([1, 512])
    print(mean.size()) # torch.Size([1, 256, 1, 1])
