from torch.utils.data import Dataset,DataLoader
import torch
import os
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import random

class Reader(Dataset):
    def __init__(self,clip_txt,transform):
        super(Reader,self).__init__()
        f = open(clip_txt, 'r')
        clip_list = f.readlines()
        f.close()
        self.clip_list = {}
        self.landmarks_list = []
        for c in clip_list:
            c = c.strip()
            d = c.split('data')
            d = d[0] + 'dataset/voxcelb1/' 'landmarks' + d[2]
            imgs = os.listdir(c)
            self.clip_list[c] = imgs
            self.landmarks_list.append(d)
        self.trans = transform
        self.flip = F.hflip
    def __getitem__(self, index):
        landmark_path = self.landmarks_list[index]
        clip_path = landmark_path.split('landmarks')
        clip_path = clip_path[0] + 'data' + clip_path[1]
        clip_imgs = self.clip_list[clip_path]
        np.random.shuffle(clip_imgs)
        imgs_for_e_path = clip_imgs[:8]
        imgs_for_training_path = clip_imgs[8:]
        imgs_for_e = []
        landmarks_for_e = []
        imgs_for_training = []
        landmarks_for_training = []
        for p in imgs_for_e_path:
            img = Image.open(os.path.join(clip_path,p)).convert('RGB')
            landmark = Image.open(os.path.join(landmark_path,p)).convert('RGB')
            is_flip = random.random() > 0.5
            if is_flip:
                img = self.flip(img)
                landmark = self.flip(landmark)
            img = self.trans(img)
            landmark = self.trans(landmark)
            imgs_for_e.append(img)
            landmarks_for_e.append(landmark)
        for p in imgs_for_training_path:
            img = Image.open(os.path.join(clip_path, p)).convert('RGB')
            landmark = Image.open(os.path.join(landmark_path, p)).convert('RGB')
            is_flip = random.random() > 0.5
            if is_flip:
                img = self.flip(img)
                landmark = self.flip(landmark)
            img = self.trans(img)
            landmark = self.trans(landmark)
            imgs_for_training.append(img)
            landmarks_for_training.append(landmark)
        return {'imgs_e':imgs_for_e,
                'landmarks_e':landmarks_for_e,
                'imgs_training':imgs_for_training,
                'landmarks_training':landmarks_for_training
                }
    def __len__(self):
        return len(self.landmarks_list)

def get_loader(clip_txt,batchsize,num_workers):
    trans = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    dataset = Reader(clip_txt,trans)
    loader = DataLoader(dataset,batchsize,pin_memory=True,drop_last=True,num_workers=num_workers)
    return loader, len(dataset)
if __name__=='__main__':
    clip_txt = 'video_clips_path.txt'
    batchsize = 1
    num_workers = 0
    loader,_ = get_loader(clip_txt,batchsize,num_workers)
    for d in loader:
        for k, v in d.items():
            print(k)
            print(type(v))
            print(type(v[0]))
            print(v[0].size())  # torch.Size([1, 3, 256, 256])

            print('**********')
