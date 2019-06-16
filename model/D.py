import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from model.resblocks import Block
from model.resblocks import OptimizedBlock


class D(nn.Module):

    def __init__(self,input_nc=6 ,num_features=64, num_classes=0, activation=F.relu):
        super(D, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        if num_classes == None:
            raise  ValueError('no given num_classes')
        self.block1 = OptimizedBlock(input_nc, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 8,
                            activation=activation, downsample=True)  # 8


        self.w_0 = nn.Parameter(torch.Tensor(1,num_features * 8))
        nn.init.xavier_normal_(self.w_0.data)
        self.b = nn.Parameter(torch.zeros(1))

        if num_classes > 0:
            self.W = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 8))

    def _initialize(self):
        optional_W = getattr(self, 'W', None)
        if optional_W is not None:
            init.xavier_uniform_(optional_W.weight.data)


    def forward(self, x, y):  # 形如[1,23,45,..] 是视频序列，范围在0 到 num_class-1
        # h = x
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        h5 = self.block5(h4)
        h5 = self.activation(h5)
        # Global pooling
        v_loss = torch.sum(h5, dim=(2, 3)) # B,C
        v_loss = torch.sum(v_loss * (self.W(y) + self.w_0.squeeze()),dim=1) + self.b  # ([B])
        v_loss = torch.mean(v_loss)
        return [v_loss, h1, h2, h3, h4, h5,self.W(y)]


if __name__=='__main__':
    model = D(num_classes=100).cuda()
    y = torch.ones([]).long().cuda()
    data = torch.randn(1,6,256,256).cuda()
    output = model(data,y)
    v_loss  = output[0]
    print(v_loss.size())  # torch.Size([])
    w = output[-1]
    print(w.size())       # torch.Size([8, 512])
