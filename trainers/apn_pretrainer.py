import torch
import numpy as np
from utils.meters import AverageMeter
import time

class APNpretrainer:
    def __init__(self,opt, model, optimizer, criterion):
        self.opt=opt
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion

    def train(self,epoch,data_loader):
        #省略summary_writer
        self.model.train()
        #计时
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        start = time.time()

        for i,inputs in enumerate(data_loader):
            imgs=self._parse_data(input)
            p2_out,attens=self.model(imgs)
            t_attens=self._get_xyl(p2_out)
            loss=self.criterion(attens,t_attens)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(loss.item())
            start = time.time()
            #输出
            if (i+1)%self.opt.print_freq==0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))

        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
        print()

    # 计算注意力区域的坐标和半边长
    def _get_xyl(apn_input):
        '''
        :param apn_input: 传入apn网络的输入，shape=(N,512,112,112) p2_out
        :return: 坐标和半边长
        '''
        N = apn_input.shape[0]
        chanels = apn_input.shape[1]
        in_size = apn_input.shape[2]
        sum_sum = torch.zeros((in_size, in_size))
        xyl = []
        for n in range(N):
            for i in range(chanels):
                sum_sum += apn_input[n, i, :, :]
            sum_sum = sum_sum / chanels
            # print(sum_sum)
            # ave = (torch.sum(sum_sum) / (width * width))
            # sum_sum[sum_sum >= ave] = 1
            # sum_sum[sum_sum < ave] = 0
            # for i in range(len(sum_sum)):
            #     print('[',end='')
            #     for j in range(len(sum_sum)):
            #         print(sum_sum[i][j].item(),end='')
            #         print(',',end='')
            #     print(']')

            # reference macnn
            columns_max_value = torch.max(sum_sum, dim=0)[0]
            rows_max_value = torch.max(sum_sum, dim=1)[0]
            left, top, right, down = 0, 0, in_size - 1, in_size - 1
            base = torch.max(sum_sum) * 0.65  # 设置为0.65
            for i in range(in_size):
                if columns_max_value[i] > base:
                    left = i
                    break
            for i in range(in_size - 1, -1, -1):
                if columns_max_value[i] > base:
                    right = i
                    break
            for j in range(in_size):
                if rows_max_value[j] > base:
                    top = j
                    break
            for j in range(in_size - 1, -1, -1):
                if rows_max_value[j] > base:
                    down = j
                    break
            x = (left + right) / 2
            y = (top + down) / 2
            l = np.maximum((right - left), (down - top)) / 2
            # 设置l的范围
            if l > in_size * 3 / 8: l = in_size * 3 / 8
            if l < in_size / 12: l = in_size / 12
            # 映射到原图
            x = x * 448 / in_size
            y = y * 448 / in_size
            l = l * 448 / in_size
            xyl.append((x, y, l))
        tensor_xyl = torch.tensor(xyl)
        return tensor_xyl

    def _parse_data(self,inputs):
        imgs,_=inputs
        imgs=imgs.cuda()
        return imgs
