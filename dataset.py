# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/19 11:33
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
import os
import xml.etree.ElementTree as ET
import numpy as np

class MyDataset(Dataset):
    def __init__(self,root='../datasets/Images',train_or_val='train',transforms=None):
        super(MyDataset, self).__init__()
        with open(root+'/'+train_or_val+'.txt',"r") as f:
            self.images=f.readlines()
        self.images=[root+'/'+i.replace('\n','') for i in self.images]
        self.transforms=transforms
        self.class_dict=os.listdir(root)
        self.class_dict.remove('train.txt')
        self.class_dict.remove('val.txt')
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        path=self.images[item]
        # BGR输入
        image=cv2.imread(path)
        if image is None:
            # 有些gif结尾的jpg实际上是gif cv imread 打不开
            # print('?? {} is not found'.format(path))
            return None,None
        image=self.transforms(image)
        label=self.class_dict.index(path.split('/')[3])
        return image,label

class VocDataset(Dataset):
    def __init__(self,root='../datasets/VOC2012',train_or_val='train',transforms=None):
        super(VocDataset, self).__init__()
        with open(root+'/'+train_or_val+'.txt',"r") as f:
            self.images_xml=f.readlines()
        self.root=root
        self.images_xml=[root+'/Annotations/'+i.replace('\n','')+'.xml' for i in self.images_xml]
        self.transforms=transforms
        self.class_dict= ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus','car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike','person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    def __len__(self):
        return len(self.images_xml)
    def __getitem__(self, item):
        xml_path=self.images_xml[item]
        tree = ET.parse(xml_path)
        root = tree.getroot()

        path=self.root+'/JPEGImages/'+root.find('filename').text
        multi_cls_lab = torch.from_numpy(np.zeros((20), np.float32))
        # BGR输入
        image=cv2.imread(path)
        if image is None:
            # 有些gif结尾的jpg实际上是gif cv imread 打不开
            # print('?? {} is not found'.format(path))
            return None,None
        image=self.transforms(image)
        for obj in root.findall('object'):
            name=obj.find('name').text
            multi_cls_lab[self.class_dict.index(name)]=1
        return image,multi_cls_lab