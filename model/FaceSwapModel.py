from model.D import D
from model.G import G
from model.E import E
import numpy as np
import torch
from torch.nn import Module
from torch import nn
from model.loss import VGGLoss,VGGFaceLoss,CNTLoss,AdvLoss,MCHLoss,DLoss
# divide whole model into E and G&D
class GDModel(Module):
    def __init__(self,num_classes):
        super(GDModel, self).__init__()
        self.g = G(input_nc=3)
        self.d = D(num_classes=num_classes)
        self.cntloss = CNTLoss()
        self.advloss = AdvLoss()
        self.mchloss = MCHLoss()
        self.dloss = DLoss()


    def g_forward(self,landmark):
        return self.g(landmark)
    def d_forward(self,fake_img,real_image,y):
        fake_info, real_info = self.d(fake_img,y), self.d(real_image,y)
        return fake_info, real_info
    def cal_cnt_loss(self,fake_image, real_image):
        return self.cntloss(fake_image,real_image)
    def cal_adv_loss(self,fake_info,real_info):  # include FM loss
        fake_v_loss = fake_info[0]
        fake_v_features = fake_info[1:6]
        real_v_features = real_info[1:6]
        return self.advloss(fake_v_features,real_v_features,fake_v_loss)
    def cal_mch_loss(self,fake_info):
        w = fake_info[6]
        return self.mchloss(w,self.e)
    def cal_d_loss(self,fake_info, real_info):
        fake_v_loss = fake_info[0]
        reak_v_loss = real_info[0]
        return self.dloss(reak_v_loss,fake_v_loss)
    def update_GDModel(self,mean_y,std_y,e):
        self.e = e
        self.g.update_adain(mean_y,std_y)

    def for_test_inference(self,landmark,y,x):
        x_landmark = torch.cat((landmark,x),1)
        fake_image = self.g_forward(landmark)   # from -1 to 1
        fake_landmark = torch.cat((landmark,fake_image),1)
        fake_info,real_info = self.d_forward(fake_landmark,x_landmark,y)
        g_loss = self.cal_cnt_loss(fake_image,x) + self.cal_adv_loss(fake_info,real_info) +\
            self.cal_mch_loss(fake_info)
        d_loss = self.cal_d_loss(fake_info,real_info)
        print(g_loss.size())
        print(d_loss.size())

    def forward(self, landmark,y,x):
        x_landmark = torch.cat((landmark,x),1)
        fake_image = self.g_forward(landmark)   # from -1 to 1
        fake_landmark = torch.cat((landmark,fake_image),1)
        fake_info,real_info = self.d_forward(fake_landmark,x_landmark,y)
        g_loss = self.cal_cnt_loss(fake_image,x) + self.cal_adv_loss(fake_info,real_info) +\
            self.cal_mch_loss(fake_info)
        d_loss = self.cal_d_loss(fake_info,real_info)
        return fake_image,g_loss,d_loss


if __name__=='__main__':
    landmark = torch.randn(2, 3, 224, 224).cuda()
    y = torch.LongTensor(np.random.randint(0, 50, size=[2])).cuda()
    x = torch.randn(2, 3, 224, 224).cuda()
    x_landmark = torch.cat((landmark, x), 1)
    e_net = E().cuda()
    e, mean_y, std_y = e_net(x_landmark)
    model = GDModel(num_classes=30,).cuda()
    model.update_GDModel(mean_y,std_y,e)
    model.for_test_inference(landmark,y,x)


