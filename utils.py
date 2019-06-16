from torch.optim import Adam
from torchvision.models import vgg16

def update_learning_rate(optim,lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr


if __name__=='__main__':
    model = vgg16()
    optim = Adam(model.parameters(),lr=3)
    print(optim.param_groups[0]['lr'])
    update_learning_rate(optim,0.001)
    print(optim.param_groups[0]['lr'])