import torch
import numpy as np
import os
from model.E import E
from model.FaceSwapModel import GDModel
from dataset.reader import get_loader
from torch.optim import Adam
from utils import update_learning_rate
from torch import nn
import visdom
import cv2

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
# vis = visdom.Visdom()

clip_txt = './dataset/video_clips_path.txt'
batchsize = 1
num_workers = 4
epoches = 60
loader, num_classes = get_loader(clip_txt,batchsize,num_workers)
E = E().cuda()
E.train()
model = GDModel(num_classes).cuda()
model.train()
for m in model.cntloss.vggfaceloss.vggface.modules():
    if isinstance(m,nn.BatchNorm2d):
        m.eval()

# 涉及到batchnorm 在training阶段，要计算batch的均值和标准差，使得batchsize不能为1，但是eval的话，又没法计算梯度
# model.cntloss.vggfaceloss.vggface.eval()
# model.cntloss.vggloss.vgg.eval()
#

# mutilple GPUs
# E = torch.nn.DataParallel(E,)
# model = torch.nn.DataParallel(model)


# define optim
g_e_parameters = list(E.parameters())
g_e_parameters += list(model.g.parameters())

lr = 2e-4
g_e_optim = Adam(g_e_parameters,lr = lr ,betas=(0.5,0.999) )
d_optim = Adam(model.d.parameters(),lr = 5 * lr,betas=(0.5,0.999))

global_step = 0
for e in range(epoches):
    current_clip_number = 0
    print('current epoch is %d'%epoches)
    for d in loader:
        print('current_clip_number is %d' % current_clip_number)
        # calculate e, std_y, mean_y for adaptive instance norm
        data_for_e = d['imgs_e']
        data_for_e = torch.cat(data_for_e,0).cuda()
        landmark_for_e = d['landmarks_e']
        landmark_for_e = torch.cat(landmark_for_e,0).cuda()
        batch_data = d['imgs_training']
        batch_data = torch.cat(batch_data,0)
        batch_landmark = d['landmarks_training']
        batch_landmark = torch.cat(batch_landmark,0)

        # e,mean_y,std_y = E(torch.cat((data_for_e,landmark_for_e),1))
        # model.update_GDModel(mean_y,std_y,e)

        # print(data_for_e.size())
        # print(landmark_for_e.size())
        for b,l in zip(batch_data,batch_landmark):  # b and l are 3-dim tensors
            
            e, mean_y, std_y = E(torch.cat((data_for_e, landmark_for_e), 1))
            model.update_GDModel(mean_y, std_y, e)

            global_step += 1
            b = b.unsqueeze(0).cuda()
            l = l.unsqueeze(0).cuda()
            y = torch.tensor(current_clip_number).long().cuda()
            fake_img, g_loss, d_loss = model(l,y,b)

            model.cntloss.vggfaceloss.vggface.zero_grad()
            model.cntloss.vggloss.vgg.zero_grad()
            g_e_optim.zero_grad()
            g_loss.backward(retain_graph=True)
            g_e_optim.step()


            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            if global_step%100==0:
                fake_img = np.transpose(np.uint8((fake_img.cpu().data.numpy()[0]/2.0 + 0.5)*255),[1,2,0])
                b_ = np.transpose(np.uint8((b.cpu().data.numpy()[0]/2.0 + 0.5)*255),[1,2,0])
                l_ = np.transpose(np.uint8((l.cpu().data.numpy()[0]/2.0 + 0.5)*255),[1,2,0])
                # temp = np.stack((fake_img,b_,l_))  # 3, 3 ,256,256
                temp = np.concatenate((fake_img[:,:,::-1],b_[:,:,::-1],l_[:,:,::-1]),axis=1)
                cv2.imwrite('./training_visual/temp_fake_gt_landmark_%d.jpg'%global_step,temp)
            # vis.images(temp,nrow=1,win='temp_results')

            print('***************')
        current_clip_number += 1
        if global_step % 50 == 0:
            saved = {'e':E.state_dict(),
                     'g_d': model.state_dict()}
            torch.save(saved,'./saved_models/e_g_d%d.pth'%global_step)
    if (e+1)%10 ==0:
        lr = lr/2.0
        update_learning_rate(g_e_optim,lr)
        update_learning_rate(d_optim,5*lr)
