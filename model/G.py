import torch
from torch.nn.utils import spectral_norm
from torch.nn import Module
from torch import nn
from torch.nn import init
import math

class G(Module):
    def __init__(self,input_nc,):
        super(G,self).__init__()
        activation = nn.ReLU()

        model = [nn.ReflectionPad2d(4),
                 spectral_norm(nn.Conv2d(input_nc,128,9,1,bias=False)),
                 nn.InstanceNorm2d(128,affine=True),
                 activation]
        model = nn.Sequential(*model)

        # downsampling
        in_nc = 128
        out_nc = 2* in_nc    # 256, 256
        for i in range(2):
            model.add_module('res_down%d'% i,Residual_Block(in_nc,out_nc))
            in_nc = out_nc
        # residual same resolution
        for i in range(4):
            model.add_module('res_iden%d'% i, Residual_Ident(in_nc,256,))

        # upsampling
        for i in range(2):
            model.add_module('up%d'%i, Up(in_nc,in_nc))
            # in_nc = in_nc//2
        # reduce dimension from 64 to 3
        model.add_module('head',nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(256,128,3,1)),
            activation,

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(128, 64, 3, 1)),
            activation,

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(64, 3, 3, 1)),
            nn.Tanh()   # -1 ,1

        ))
        self.model = model
    def forward(self,x):
        x= self.model(x)
        # print('g mean : ',x.mean())
        return x
    def update_adain(self,mean_y,std_y):
        for m in self.modules():
            if isinstance(m,Adain):
                m.update_mean_std(mean_y,std_y)


class Residual_Block(Module):
    def __init__(self,input_nc,output_nc):
        super(Residual_Block, self).__init__()
        activation = nn.ReLU(True)
        self.left = nn.Sequential(*[spectral_norm(nn.Conv2d(input_nc,output_nc,1,1,padding=0,bias=False)),
                                     nn.InstanceNorm2d(output_nc,affine=True),
                                     activation,
                                     nn.AvgPool2d(3, 2, padding=1)
                                     ])
        self.right = nn.Sequential(*[
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(input_nc,output_nc,3,1,padding=0,bias=False)),
            nn.InstanceNorm2d(output_nc,affine=True),
            activation,
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(output_nc, output_nc, 3, 1, padding=0, bias=False)),
            nn.InstanceNorm2d(output_nc,affine=True),
            activation,
            nn.AvgPool2d(3, 2, padding=1)
        ])

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                raise ValueError('No this layer init way')

    def forward(self, x):
        x1 = self.right(x)
        x2 = self.left(x)
        return x1+x2

class Residual_Ident(Module):
    def __init__(self,in_nc,out_nc,):
        super(Residual_Ident,self).__init__()
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(1),
                 spectral_norm(nn.Conv2d(in_nc,out_nc,3,stride=1,padding=0,bias=False)),
                 nn.InstanceNorm2d(out_nc,affine=False),
                 Adain(),
                 activation
                 ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class Adain(Module):
    def __init__(self,):
        super(Adain,self).__init__()

    def forward(self, x):
        size = x.size()
        x = x*self.std_y.expand(size) + self.mean_y.expand(size)
        return x
    def update_mean_std(self,mean_y,std_y):   # be used before forward
        self.mean_y = mean_y
        self.std_y = std_y

class Up(Module):
    def __init__(self,in_nc,out_nc):
        super(Up, self).__init__()
        activation = nn.ReLU(True)
        model = [
                nn.Upsample(scale_factor=2,mode='bilinear'),
                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(in_nc,out_nc,3,1,bias=False)),
                nn.InstanceNorm2d(out_nc,affine=False),
                Adain(),
                activation
                 ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        x = self.model(x)
        return x

if __name__=='__main__':
    std_y = torch.randn(2,256,1,1).cuda()
    x = torch.randn(2,6,224,224).cuda()
    mean_y = torch.zeros(2,256,1,1).cuda()

    model = G(6).cuda()
    model.update_adain(mean_y,std_y)
    out = model(x)
    print(out.size())   # torch.Size([2, 3, 224, 224])



    # for m in model.modules():
    #     if isinstance(m,Adain):
    #         m.update_mean_std(mean_y,std_y)
    # for m in model.children():
    #     for c in m.modules():
    #         if isinstance(c,Adain):
    #             # print(c.mean_y==mean_y)
    #             # print(c.std_y==std_y)
    #             print('*********') # 6次