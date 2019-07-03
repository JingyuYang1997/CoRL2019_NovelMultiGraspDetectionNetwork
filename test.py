import torch
from models import Bottleneck, HourGlass
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode",default='test')
args = parser.parse_args()
print(args)

save_path = './MultiGraspDataset/valid/{}/compare/'.format(args.mode)
if not os.path.exists(save_path):
    os.mkdir(save_path)

net = HourGlass(Bottleneck,num_classes=10)
net.load_state_dict(torch.load('checkpoints/ep_40.pt'))
net.cpu()
image_files = glob.glob('./MultiGraspDataset/valid/{}/src_images/*.png'.format(args.mode))
for image_file in image_files:
    image_index = (image_file.split('/')[-1]).split('_')[0]
    image  = cv2.imread(image_file)
    image = cv2.resize(image,(256,256))
    input_image = image[...,::-1]
    input_image = torch.FloatTensor(np.transpose(input_image[None, :, :, :].astype('float32')/255.0 , [0, 3, 1, 2]))
    output_hm = net(input_image)
    hm = []
    for j in range(output_hm.shape[1]):
        heatmap = output_hm[0, j, :, :].detach().numpy() * 255
        heatmap = np.clip(heatmap, 0, 255)
        hm.append(heatmap)
    hm_overall = sum(hm)
    hm_overall = np.clip(hm_overall, 0, 255)
    hm_overall = hm_overall.astype('uint8')
    plt.subplot(121)
    plt.imshow(image[...,::-1])
    plt.subplot(122)
    plt.imshow(hm_overall)
    plt.savefig('./MultiGraspDataset/test_images/compare/{}_cp.png'.format(image_index))
    plt.close()


