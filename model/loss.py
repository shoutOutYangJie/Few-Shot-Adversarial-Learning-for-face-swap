from torchvision import models
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torchvision.transforms import transforms
from model.vgg_face import vgg_m_face_bn_dag

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # self.zero_grad()
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss,self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss(size_average=True)
        self.weight = [0.01] *5
    def forward(self, x,y):
        x_vgg, y_vgg = self.vgg(x),self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss = self.weight[i]/float(len(x_vgg)) * self.criterion(x_vgg[i],y_vgg[i].detach()) + loss
        return loss
class VGGFaceLoss(nn.Module):
    def __init__(self):
        super(VGGFaceLoss,self).__init__()
        self.vggface = vgg_m_face_bn_dag('./pretrained/vgg_m_face_bn_dag.pth')
        # self.vggface = vgg_m_face_bn_dag(None)
        self.criterion = nn.L1Loss(size_average=True)
        self.weights = [0.002] * 5
    def forward(self, x,y):
        x_vggface, y_vggface = self.vggface(x), self.vggface(y)
        loss = 0
        for i in range(len(x_vggface)):
            loss += self.weights[i]/5.0 * self.criterion(x_vggface[i],y_vggface[i].detach())
        return loss

class CNTLoss(nn.Module):
    def __init__(self):
        super(CNTLoss, self).__init__()
        self.vggloss = VGGLoss()
        self.vggfaceloss = VGGFaceLoss()
        self.vgg_preprocessed = transforms.Normalize([0.485,0.456,0.406],
                                                     [0.229,0.224,0.225])
    def forward(self,x,y):
        x = (x+1)*127.5
        y = (y+1)*127.5
        x1 = self.vgg_preprocessed(x.squeeze(0)/255.0).unsqueeze(0)
        y1 = self.vgg_preprocessed(y.squeeze(0)/255.0).unsqueeze(0)
        loss = 0.5 * (self.vggfaceloss(x,y) + self.vggloss(x1,y1))
        return loss


class AdvLoss(nn.Module):
    def __init__(self,):
        super(AdvLoss,self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def forward(self, fake_feature,real_feature,v_loss):
        fm_loss = 0
        feat_weights = 10.0 / len(fake_feature)
        for i in range(len(fake_feature)-1):
            fm_loss += feat_weights * self.criterion(fake_feature[i],real_feature[i].detach())
        return  -v_loss + fm_loss  # D_loss必须是标量才行


class MCHLoss(nn.Module):
    def __init__(self):
        super(MCHLoss,self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def forward(self, w,e):
        return 80 * self.criterion(w, e.detach())

class DLoss(nn.Module):
    def __init__(self):
        super(DLoss,self).__init__()
    def forward(self, real_vloss, fake_vloss):
        d_loss = torch.mean(torch.relu(1. - real_vloss)) +\
            torch.mean(torch.relu(1. + fake_vloss))
        return d_loss
