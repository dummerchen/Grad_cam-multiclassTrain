# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/16 13:17
# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen
# @Time : 2022/5/16 12:36
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18,vgg11
from torchvision import transforms
import torch
from torch import nn
import cv2
from utils import GradCam,GuideBackPropagation

if __name__ == '__main__':
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 10)
    params = torch.load('./weights/resnet_16.pth', map_location=device)
    model.load_state_dict(params['weights'])

    data_transforms = {
        'train': transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'val': transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }

    model.eval()

    feature_names = [model.layer4[1].conv2]
    # BGR输入
    image = cv2.imread('./Images/kitchen/int474.jpg')
    image2 = data_transforms['val'](image)
    cam = GradCam(feature_names, target_size=(224,224))
    cv2.imshow('0',image)
    res = model(torch.unsqueeze(image2[0], dim=0).to(device))
    pre = torch.argmax(res)
    print(pre)
    res[0, pre].backward()
    images, masks = cam.cal_cam(image,rgb_or_bgr=False)

    cv2.imshow('1',images[0])
    cv2.waitKey()
    cv2.destroyAllWindows()