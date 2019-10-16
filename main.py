from torch.utils.data import DataLoader
import torch
from config import opt
import pprint
from torch.backends import cudnn
from datasets.data_manager import Food101
from utils.transforms import TrainTransform
from utils.transforms import TestTransform
from datasets.data_loader import ImageData
import torch.nn as nn
import torchvision.models as models
from models.resnet import get_resnet101
from trainers.apn_pretrainer import APNpretrainer

#train
def train(**kwargs):
    opt._parse(kwargs)  # 设置程序的所有参数
    torch.manual_seed(opt.seed)
    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')
    print('initializing dataset {}'.format(opt.dataset))
    datasets = Food101()
    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageData(datasets.train, TrainTransform()), batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True, shuffle=True
    )
    evaluateloader = DataLoader(
        ImageData(datasets.val, TestTransform()), batch_size=opt.val_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')






    print('model size: {:.5f}M'.format(sum(p.numel() for p in first_cnn.parameters()) / 1e6))
    print('initializing model end ...')

    #trainer
    apn_pretrainer = APNpretrainer(opt, first_cnn, optimizer, criterion)



    def train(dataloader,model):
        datasets = Food101()
        pin_memory = True if use_gpu else False

        if use_gpu:
            # first_cnn = nn.DataParallel(first_cnn).cuda()
            first_cnn = first_cnn.cuda()

        # 损失函数
        if opt.loss_fnc == 'MSELoss':
            criterion = nn.MSELoss()
        elif opt.loss_fnc == 'CrossEntropy':
            criterion = nn.CrossEntropyLoss()

        def optim_policy(model):
            if opt.pretrained_model:
                # 返回第一层和全连阶层的权重
                needed_optim = []
                for param in model.features[0].parameters():
                    needed_optim.append(param)
                for param in model.classifier.parameters():
                    needed_optim.append(param)
                return needed_optim
            else:
                return model.parameters()

        if opt.optim == 'SGD':
            optimizer = torch.optim.SGD(optim_policy(first_cnn), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
        else:
            # 使用adma
            optimizer = torch.optim.Adam(first_cnn.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        start_epoch = opt.start_epoch

        def adjust_lr(optimizer, ep):
            if ep < 15:
                lr = 1e-4 * (ep // 5 + 1)
            else:
                lr = 6e-5
            for p in optimizer.param_groups:
                p['lr'] = lr

        best_rank1 = opt.best_rank
        best_epoch = 0
        for epoch in range(start_epoch, opt.max_epoch):
            if opt.adjust_lr:
                adjust_lr(optimizer, epoch)
            avg_loss, avg_correct = 0, 0
            first_cnn.train()
            for batch_idx, inputs in enumerate(trainloader):
                optimizer.zero_grad()
                data, target = inputs
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                try:
                    outputs = first_cnn(data)
                except RuntimeError as e:
                    import sys
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory, retrying batch', sys.stdout)
                        sys.stdout.flush()
                        for p in first_cnn.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        outputs = first_cnn(data)
                    else:
                        raise e

                loss = criterion(outputs, target)  # outputs是二维(N,C)，target是一维(N)
                loss.backward()
                optimizer.step()
                if (batch_idx + 1) % opt.print_freq == 0:
                    num = (batch_idx + 1) * len(data)
                    print('Train:  Epoch: {} batch: {} [{}/{} ({:.2f}%)]  Loss_per_image: {:.4f}'.format(
                        epoch, batch_idx + 1, num, len(trainloader.dataset), 100. * (batch_idx + 1) / len(trainloader),
                        loss.item()))
                avg_loss += loss
                pred = outputs.data.max(1, keepdim=True)[1]
                avg_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            avg_loss = avg_loss / (batch_idx + 1)
            train_loss_dic[epoch] = avg_loss
            train_acc.append(avg_correct.item() / len(trainloader.dataset))
            print('\ntrain  loss_per_image: {:.3f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                avg_loss, avg_correct, len(trainloader.dataset), 100 * avg_correct.numpy() / len(trainloader.dataset)))
    ##         #验证并保存模型
    #         if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
    #             if opt.mode=="class":
    #                 rank1=test(model,testloader)
    #             else:
    #                 print('evaluating...')
    #                 rank1=evaluate(first_cnn,evaluateloader)
    #                 print('evaluating end...')
    #                 if rank1>best_rank1:
    #                     best_rank1=rank1
    #                     best_epoch=epoch
    #                     #保存
    #                     print('saving model...')
    #                     state_dict=first_cnn.state_dict()
    #                     save_checkpoint({"state_dict":state_dict,"epoch":epoch+1},save_dir=opt.save_dir,
    #                         filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')
    #                     print("save model first_cnn successful!...")