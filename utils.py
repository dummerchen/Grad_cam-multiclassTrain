# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/16 12:36
import torch
import numpy as np
import cv2
from torch import nn
import logging
from logging.handlers import RotatingFileHandler
from torch.utils.data import Dataset,DataLoader

from torch.utils.data.dataloader import default_collate
import os

def collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据

class MyDataset(Dataset):
    def __init__(self,root='../datasets/Images',train_or_val='train',transforms=None):
        super(MyDataset, self).__init__()
        with open(root+'/'+train_or_val+'.txt',"r") as f:
            self.images=f.readlines()
        self.images=[root+'/'+i.replace('\n','') for i in self.images]
        self.transforms=transforms
        self.class_dict=os.listdir(root)

    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        path=self.images[item]
        # BGR输入
        image=cv2.imread(path)
        if image==None:
            print('{} is not found'.format(path))
            return None,None
        image=self.transforms(image)
        label=self.class_dict.index(path.split('/')[3])
        return image,label

class GuideBackPropagation:
    def __init__(self,model,model_name:list):
        self.fh=[]
        self.bh=[]
        self.model=model

        # 所有relu层都注册
        for name, module in model.features.named_children():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(self.forward_hook)
                module.register_full_backward_hook(self.backward_hook)
        # 第一个卷积层位置

        for i in model_name:
            i.register_full_backward_hook(self.first_backward_hook)

    def normalize(self,I):
        # 归一化梯度map，先归一化到 mean=0 std=1
        norm = (I - I.mean()) / I.std()
        # 把 std 重置为 0.1，让梯度map中的数值尽可能接近 0
        norm = norm * 0.1
        # 均值加 0.5，保证大部分的梯度值为正
        norm = norm + 0.5
        # 把 0，1 以外的梯度值分别设置为 0 和 1
        norm = norm.clip(0, 1)
        return norm

    def forward_hook(self,module,input,output):
        self.fh.append(output.cpu().detach().numpy())

    def first_backward_hook(self,module,grad_in,grad_out):
        # 获取第一个卷积层反向传播的权重
        # BCHW-> CHW ->h,w,c
        self.image=grad_in[0].cpu().detach()[0]
        self.image=self.image.permute(1,2,0).numpy()
        self.image=self.normalize(self.image)
        pass
    def backward_hook(self,module, grad_in, grad_out):
        a=self.fh.pop()
        a[a>0]=1
        # 反向传播 relu
        new_grad=torch.clamp(grad_out[0],min=0.0)
        # rule 是返回一个参数,返回修改后的梯度
        return (new_grad*a,)

class GradCam:
    def __init__(self,model_name:list,target_size=None):
        self.fh=[]
        self.bh=[]
        self.target_size=target_size
        for i in model_name:
            i.register_forward_hook(self.forward_hook)
            i.register_full_backward_hook(self.backward_hook)

    def forward_hook(self,module,input,output):
        self.fh.append(output.cpu().detach().numpy())

    def backward_hook(self,module, grad_in, grad_out):
        self.bh = [grad_out[0].cpu().detach().numpy()] + self.bh

    def get_rgb_image(self,image,mask,colormap: int = cv2.COLORMAP_JET,rgb_or_bgr=False,use_heatmap=True):
        if use_heatmap==True:
            mask = np.uint8(mask * 255)
            heat_map=cv2.applyColorMap(mask,colormap=colormap)
            if rgb_or_bgr==True:
                heat_map=cv2.cvtColor(heat_map,cv2.COLOR_BGR2RGB)
            heat_map=heat_map/255.
            image=image+heat_map
            image=image/np.max(image)
        else:
            mask=np.expand_dims(mask,axis=2)
            image=image*mask

        return image

    def cal_cam(self,image,colormap: int = cv2.COLORMAP_JET,rgb_or_bgr=False,use_heatmap=True):
        '''
            获得各层的 mask和image

        :param image: 输入的图片 0~1,hwc bgr 格式
        :param colormap: 热力图colormap格式
        :param rgb_or_bgr: 输入图片是否为rgb格式HWC
        :param use_heatmap: 输出热力图或蒙版图
        :return: images:[ndarray hwc bgr],masks:[ndarray hw,gray]
        '''
        self.Image=[]
        self.masks=[]
        for a, a_ in zip(self.fh, self.bh):
            # B,C,W,H -> B,C,1,1

            alpha = np.mean(a_, axis=(2,3),keepdims=True)
            # BC,1,1-> B,1,1
            mat = np.sum(alpha*a,axis=1)
            # relu
            mat[mat<0]=0
            # 转换到0-1
            mat=(mat-np.min(mat))/(np.max(mat)+1e-7)
            if self.target_size!=None:
                # targetsize W,H
                mask=cv2.resize(mat,self.target_size)
            else:
                mask=mat
            image=self.get_rgb_image(image,mask,colormap=colormap,rgb_or_bgr=rgb_or_bgr,use_heatmap=use_heatmap)
            self.masks.append(mask)
            self.Image.append(image)
        return self.Image,self.masks

class Logger():
    def __init__(self,file_path,level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=level)

        handler = RotatingFileHandler(file_path,maxBytes=1024,backupCount=3)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def info(self,info):
        return self.logger.info(info)

    def debug(self,info):
        return self.logger.info(info)

    def warning(self,info):
        return self.logger.warning(info)

    def error(self,info):
        return self.logger.error(info)
