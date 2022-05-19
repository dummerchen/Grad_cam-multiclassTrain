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
import numpy as np
import cv2
from utils import GradCam,GuideBackPropagation
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def predict_cifar10(model):
    model.fc = torch.nn.Linear(512, 10)
    model.eval()
    params = torch.load('./weights/resnet_26.pth', map_location=device)
    model.load_state_dict(params['weights'])
    print(model.conv1)
    gbp = GuideBackPropagation(model, model_name=[model.conv1])
    # cam = GradCam([model.layer4[1].conv2])
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }
    val_datasets=CIFAR10(root='./',download=False,train=False,transform=data_transforms['train'])
    # 3,10
    image=val_datasets.data[10]
    label=val_datasets.targets[10]
    image2=data_transforms['val'](image)
    res=model(torch.unsqueeze(image2,0).requires_grad_())
    pre=torch.argmax(res)
    print(pre,label)
    res[0,pre].backward()

    # images, masks = cam.cal_cam(image, rgb_or_bgr=True)
    resimage=gbp.image
    image=cv2.resize(image,(224,224))
    resimage=cv2.resize(resimage,(224,224))
    cv2.imshow('0',image)
    cv2.imshow('1', resimage)
    cv2.waitKey()
    cv2.destroyAllWindows()

def predict_indoor(model):
    model.fc = torch.nn.Linear(512, 69)
    params = torch.load('./weights/resnet_indoor_31.pth', map_location=device)
    model.load_state_dict(params['weights'])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }

    model.eval()

    feature_names = [model.layer4[1].conv2]
    # BGR输入
    image = cv2.imread('../datasets/Images/kitchen/int474.jpg')
    image2 = data_transforms['val'](image)
    cam = GradCam(feature_names, target_size=(224, 224))
    cv2.imshow('0', image)
    res = model(torch.unsqueeze(image2, dim=0).to(device))
    print(res)
    for i in range(0, 69):
        res = model(torch.unsqueeze(image2, dim=0).to(device))
        res[0, i].backward()
        images, masks = cam.cal_cam(image2.cpu().detach().permute(1, 2, 0).numpy(), rgb_or_bgr=False)

        cv2.imshow(str(i), masks[0])
        cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

    model = resnet18(pretrained=True)
    predict_cifar10(model)
