# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2022/5/16 13:17
# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen
# @Time : 2022/5/16 12:36
from torchvision.models import resnet18,vgg11,resnet50
from torchvision import transforms
import torch
import numpy as np
import cv2
from utils import GradCam,GuideBackPropagation
def predict_voc(model):
    model.fc = torch.nn.Linear(2048, 20)
    model.eval()
    params = torch.load('./weights/resnet_voc_50.pth', map_location=device)
    model.load_state_dict(params['weights'])
    print(model.conv1)
    gbp = GuideBackPropagation(model, model_name=[model.conv1])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(256),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }
    name='2007_002400.jpg'
    # gbp
    image = cv2.imread('../datasets/VOC2012/JPEGImages/'+name)
    image2 = data_transforms['val'](image)
    c,h,w=image2.shape
    new_image=image2.permute(1,2,0).numpy()
    res = model(torch.unsqueeze(image2, 0).requires_grad_())
    pre = torch.argmax(res)

    print(pre)
    res[0, pre].backward()

    resimage = gbp.image
    cv2.imshow('0', new_image)
    cv2.imwrite('./result/raw_'+name,image)
    cv2.imshow('1', resimage)
    cv2.imwrite('./result/gbp_'+name,resimage*255)
    # cam
    cam = GradCam([model.layer4[1].conv2],target_size=(w,h))
    res = model(torch.unsqueeze(image2, 0))
    pre = torch.argmax(res)
    print(pre)
    # 12 是马 16是羊 9 是牛 11 是狗
    pre=2
    res[0,pre].backward()
    images, masks = cam.cal_cam(image/255., rgb_or_bgr=True)

    cv2.imshow('2', images[0])
    cv2.imshow('3', masks[0])
    cv2.imwrite('./result/heatmap_'+str(int(pre))+'_'+name,images[0]*255)
    cv2.imwrite('./result/mask_'+str(int(pre))+'_'+name,masks[0]*255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

    model = resnet50(pretrained=True)
    predict_voc(model)