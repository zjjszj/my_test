import torch
import torch.nn as nn
import cv2
import numpy as np



class CropAmplificate(nn.Module):

    def __init__(self):
        super(CropAmplificate,self).__init__()

    def forward(self, x,a,b,c):
        CropAndAmplification.forward(x,a,b,c)

class CropAndAmplification(torch.autograd.Function):

    # 输入448图像、坐标、半边长，输出裁剪之后的图像
    def crop_amplificate(self,img, a, b, c):  # img:Image
        x_tl = a - c
        y_tl = b - c
        x_br = a + c
        y_br = b + c
        x_tl = np.maximum(0, np.int(x_tl))
        y_tl = np.maximum(0, np.int(y_tl))
        x_br = np.minimum(448, np.int(x_br))
        y_br = np.minimum(np.int(y_br), 448)
        # print('x_tl,y_tl,x_br,y_br=',x_tl,y_tl,x_br,y_br)
        N=img.shape[0]
        roi = (x_tl, y_tl, x_br, y_br)
        img_cropped = img.crop(roi)
        # img_cropped.show()
        # 放大
        cv2_img_cropped = cv2.cvtColor(np.asarray(img_cropped), cv2.COLOR_RGB2BGR)
        cv2.resize(cv2_img_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('cv2',cv2_img_cropped)
        # cv2.waitKey(0)
        return cv2_img_cropped

    @staticmethod
    def forward(self, x,a,b,c):
        self.crop_amplificate(x,a,b,c)

    @staticmethod
    def backward(self,img_crop_amplificates):
        x_tl = a - c
        y_tl = b - c
        x_br = a + c
        y_br = b + c
        x_tl = np.maximum(0, np.int(x_tl));
        y_tl = np.maximum(0, np.int(y_tl));
        x_br = np.minimum(448, np.int(x_br));
        y_br = np.minimum(np.int(y_br), 448)

        # print('x_tl,y_tl,x_br,y_br=',x_tl,y_tl,x_br,y_br)
        def h(x, k=10):
            return (1 / (1 + np.exp(-1 * k * x)))

        def M(x, y):
            return (h(x - x_tl) - h(x - x_br)) * (h(y - y_tl) - h(y - y_br))

        unit = torch.stack([torch.arange(0, 448)] * 448)
        x = torch.stack([unit] * 3)
        y = torch.stack([unit.t()] * 3)
        N = img.shape[0]
        # print(torch.round(M(x,y)[0][y_tl:y_br]))
        # M(x,y)[M(x,y)>=0.1]=1     #还能反向传播吗？
        # M(x,y)[M(x,y)<0.1]=0
        Xatt_nonzero = torch.ones([1, 3, y_br - y_tl, x_br - x_tl])
        for i in range(N):
            Xatt = img[i, :, :, :] * torch.round(M(x, y).float())  # 3d
            Xatt = Xatt.reshape(1, 3, 448, 448)

            print('Xatt_nonzero.shape=', Xatt.shape)
            Xatt = F.interpolate(Xatt, 224, mode='bilinear')
            plt.imshow(Xatt.reshape(3, 224, 224).numpy()[0, :, :])
            plt.show()

