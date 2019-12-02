from __future__ import division

from utils.utils import *
from utils.datasets import *
import cv2
from PIL import Image
import torch
from torchvision import transforms


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def yolo_prediction(model, device, image,class_names):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    imgs = transforms.ToTensor()(Image.fromarray(image))
    c, h, w = imgs.shape
    img_sacle = [w / 416, h / 416, w / 416, h / 416]
    imgs = resize(imgs, 416)
    imgs = imgs.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(imgs)
        outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.45)

    # print(outputs)
    objects = []
    try:
        outputs = outputs[0].cpu().data
        for i, output in enumerate(outputs):
            item = []
            item.append(class_names[int(output[-1])])
            item.append(float(output[4]))
            box = [int(value * img_sacle[i]) for i, value in enumerate(output[:4])]
            x1,y1,x2,y2 = box
            x = int((x2+x1)/2)
            y = int((y1+y2)/2)
            w = x2-x1
            h = y2-y1
            item.append([x,y,w,h])
            objects.append(item)
    except:
        pass
    return objects



