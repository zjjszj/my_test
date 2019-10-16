#resnet + fpn

#使用预定义模型
import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as T
import math
import torch.nn.functional as F
from config import opt

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)
        p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
        p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)
        return p2_out

def get_resnet101(classmodel,trained=False):
    '''
    :param model: pretrained resnet101
    :param trained:classmodel trained in cache directory
    :return:resnet101
    '''
    # 修改全连阶层
    classmodel.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.Linear(1024, 101)
    )
    # 全连阶层参数赋值
    for m in classmodel.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        #冻结未改变层的参数
#         elif isinstance(m,nn.Conv2d):
#             for p in m.parameters():
#                 p.requires_grad=False
#         elif isinstance(m,nn.BatchNorm2d):
#             for p in m.parameters():
#                 p.requires_grad=False
    if trained:
        state_dict=torch.load(opt.trained_path)
        classmodel.load_state_dict(state_dict)
    return classmodel

def get_partclassmodel():
    pass
