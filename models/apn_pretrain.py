import torch.nn as nn
from models.resnet import get_resnet101,FPN,get_partclassmodel
from torchvision.models.resnet import resnet101


class APNPretrainModel(nn.Module):
    def __init__(self):
        super(APNPretrainModel,self).__init__()
        self.classmodel=get_partclassmodel()
        self.c1 = nn.Sequential(
            self.classmodel.conv1,
            self.classmodel.bn1,
            self.classmodel.relu,
            self.classmodel.maxpool
        )
        self.c2 = self.classmodel.layer1
        self.c3 = self.classmodel.layer2
        self.c4 = self.classmodel.layer3
        self.c5 = self.classmodel.layer4
        self.fpn=FPN(self.c1,self.c2,self.c3,self.c4,self.c5,512)



    def forward(self,x):

        p2_out=FPN(self.classmodel(x))







        
