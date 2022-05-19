# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/16 12:36
import argparse
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18,resnet50
from torch.utils.data import Dataset,DataLoader
import torch
import os
from utils import GradCam,Logger,collate_fn
from dataset import MyDataset,VocDataset
import torch.backends.cudnn as cudnn
from torch.utils import tensorboard
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix



def main(opts):
    batchsize=opts.batchsize
    start_epoch=0
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
    print('using device:{}'.format(device))
    if not os.path.exists(opts.logs_dir):
        os.mkdir(opts.logs_dir)

    cudnn.benchmark=True
    torch.manual_seed(opts.seed)
    dir_name=os.path.join(opts.logs_dir,'resnet_'+time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())))
    writer=tensorboard.SummaryWriter(log_dir=dir_name)
    logger=Logger('logs/log.txt')
    if opts.workers==None:
        workers=min([os.cpu_count(), opts.batchsize if opts.batchsize > 1 else 0, 8])
    else:
        workers=opts.workers
    print('use workers:{}'.format(workers))

    data_transforms={
        'train':transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'val':transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.RandomResizedCrop(256),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }
    train_dataset = VocDataset(root=opts.root_dir,train_or_val='train', transforms=data_transforms['train'])
    val_dataset = VocDataset(root=opts.root_dir,train_or_val='val',transforms=data_transforms['val'])
    # train_dataset = MyDataset(root=opts.root_dir,train_or_val='train', transforms=data_transforms['train'])
    # val_dataset = MyDataset(root=opts.root_dir,train_or_val='val',transforms=data_transforms['val'])

    # train_dataset=CIFAR10(root='./', download=False, train=True, transform=data_transforms['train'])
    # val_dataset = CIFAR10(root='./', download=False, train=False, transform=data_transforms['train'])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_fn
    )

    model=resnet50(pretrained=True)
    model.fc=torch.nn.Linear(2048,20)
    model.to(device)

    loss_func=torch.nn.MultiLabelSoftMarginLoss()
    optimizer=torch.optim.Adam(lr=opts.learning_rate,params=model.parameters())
    if  opts.weights!=None and os.path.exists(opts.weights):
        # 加载模型
        params=torch.load(opts.weights,map_location=device)
        optimizer.load_state_dict(params['optimizer'])
        model.load_state_dict(params['weights'])
        start_epoch=params['epoch']


    for epoch in range(start_epoch+1,opts.epoch+1):
        model.train()
        train_bar=tqdm(train_dataloader)
        train_mean_loss=0
        tot=0
        for data,label in train_bar:
            data,label=data.to(device),label.to(device)
            res=model(data)
            res=torch.nn.Softmax()(res)
            loss=loss_func(res,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot+=1
            # if tot>2:
            #     break
            with torch.no_grad():
                train_mean_loss+=loss.item()
                train_bar.set_description(desc='Train Epoch[{}/{}] loss:{}'.format(epoch,opts.epoch,loss.item()))

        train_mean_loss=train_mean_loss/len(train_bar)
        writer.add_scalar('train_loss',train_mean_loss,epoch)

        logger.info('train_loss:{}\n'.format(train_mean_loss))

        if epoch%opts.save_epoch==0:
            model.eval()

            # 画图
            # data, label = next(iter(val_dataloader))
            # c,h,w=data[0].shape
            # cam = GradCam([model.layer4[2].conv3], target_size=(h,w))
            #
            # res=model(torch.unsqueeze(data[0],dim=0).to(device))
            # res = torch.nn.Sigmoid()(res)
            # pre=torch.argmax(res)
            # res[0,pre].backward()
            # images, masks = cam.cal_cam(data[0].permute(1, 2, 0).numpy())
            #
            # writer.add_images('resnet/epoch_' + str(epoch) + '_image',
            #                   torch.unsqueeze(torch.Tensor(images[0].transpose(2,0,1)),dim=0), epoch)

            val_bar=tqdm(val_dataloader)
            val_mean_loss = 0
            val_mean_acc=0
            val_mean_f1=0
            with torch.no_grad():
                for data,label in val_bar:
                    data, label = data.to(device), label.to(device)

                    res=model(data)
                    res = torch.nn.Softmax()(res)

                    pre=res.numpy()
                    pre[pre>0.5]=1
                    pre[pre<0.5]=0
                    acc=accuracy_score(label.cpu().detach().numpy(),pre)
                    f1=f1_score(label.cpu().detach().numpy(),pre,average='macro')
                    loss=loss_func(res,label)

                    val_mean_loss+=loss.item()
                    val_mean_acc+=acc
                    val_mean_f1+=f1

                    val_bar.set_description(desc='val Epoch:{} Loss:{} acc:{} f1:{}'.format(epoch,loss.item(),acc,f1))
                    # 每次只输出最后一组的验证结果
                val_mean_loss/=len(val_bar)
                val_mean_acc/=len(val_bar)
                val_mean_f1/=len(val_bar)
                writer.add_scalar('val_mean_loss',val_mean_loss,epoch)
                writer.add_scalar('val_mean_f1',val_mean_f1,epoch)
                writer.add_scalar('val_mean_acc',val_mean_acc,epoch)

                logger.info('val_mean_loss:{} mean_acc:{} mean f1:{}\n'.format(val_mean_loss, val_mean_acc,val_mean_f1))
            state_dict={
                'weights':model.state_dict(),
                'epoch':epoch,
                'optimizer':optimizer.state_dict(),
                }
            torch.save(state_dict,'./weights/resnet_voc_{}.pth'.format(epoch))
    writer.close()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--batchsize', '-bs', default=16, type=int)
    args.add_argument('--root_dir', '-rd', default='../datasets/VOC2012', type=str)
    args.add_argument('--seed', default=1314, type=int)
    args.add_argument('--weights', '-w', default=None, type=str)
    args.add_argument('--logs_dir', '-ld', default='./logs', type=str)
    args.add_argument('--learning_rate', '-lr', default=0.001, type=float)
    args.add_argument('--epoch', '-e', default=100, type=int)
    args.add_argument('--save_epoch', '-se', default=1, type=int)
    args.add_argument('--workers', '-wks', default=None, type=int)

    opts = args.parse_args()
    main(opts)