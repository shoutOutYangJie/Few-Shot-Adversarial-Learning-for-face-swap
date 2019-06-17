# Few-Shot-Adversarial-Learning-for-face-swap
This is a unofficial re-implementation of the paper "Few-Shot Adversarial Learning of Realistic Neural Talking Head Models" based on Pytorch.

# description
The paper from SAMSUNG AI lab presents a new novel and efficient method for face swap, which has amazing performance. I am a student and interested in this paper, so I try to reproduce it.
I have writen the script for getting landmarks dataset, and dataloader pipeline. I also construct the whole network including Embedding, Generator and Discriminator with loss functions. Everything is done according to the paper.
But, due to unspecific network descripted by the paper, I found that there are some mistakes in Generator, especially for the Adaptive instance normalization. The training process gets weird results, which you can look at it from "training_visual" file.
**For getting more understanding about this amazing and great work, I open source this projection and invite more people who are interested in it to become contributor.** If someone reproduces it successfully in the future, please tell me. Thanks!

# how to use

1. get landmarks information.

you should download the [VoxCeleb, about 36GB](http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/dense-face-frames.tar.gz). And after unzip the dataset, please change the name of root path of the dataset as "voxcelb1", or you can adapt "./face-alignment_projection/get_landmarks.py" for your envirnoment.
> python get_landmarks.py

For test you shouldn't preprocess all data in the VoxCeleb dataset, because it's time-consuming. I just use about 200 video clips. The two files are generated. "video_clips_path.txt" records all clips which have been preprocessed. "video_landmarks_path.txt" records landmarks information path.

2. download VGGFace weights for perceptual loss.

you can download from [here](http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_m_face_bn_dag.py) [on this website](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html).
make sure that put the weight file under the "./pretrained" file path.

3. start to train the whole network.

> python ./train.py

For each 100 iteartion, you can look at temp results including fake image, GT and landmarks under the "./training_visual" file path.

# Cite
[1] Few-Shot Adversarial Learning of Realistic Neural Talking Head Models

[2] face-alignment is from [the great work](https://github.com/1adrianb/face-alignment)
