import numpy as np
from torch.utils.data import Dataset,DataLoader
import copy
import matplotlib.pyplot as plt
import pdb
import cv2
import glob
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import math
import scipy.ndimage as nimg

class MultiDataset(Dataset):

    def __init__(self,mode='train', sigma=2.0, lenth=18):
        if mode=='train':
            dir_name = 'MultiGraspDataset/'
        else:
            dir_name = 'test/'
        self.dir_name = dir_name
        self.sigma = sigma 
        self.lenth = lenth
        self.fnames = glob.glob(self.dir_name+'images/*.png')
    
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        img = img.resize((256, 256))
        img = ToTensor()(img)
        prefix = fname.split('/')[-1].split('_')[0]
        label_fname = self.dir_name + 'label_txt/' + prefix + '.txt'
        hm_label = self.hm(label_fname, 64, 64, self.sigma, self.lenth)
        hm_label = hm_label.astype('float32')
        hm_label = hm_label / 255.0
        return img, hm_label

    def hm(self,fname, H, W, sigma, lenth):
        label = np.loadtxt(fname)
        hm_label = np.zeros((10, H, W))
        for i, item in enumerate(label):
            img = np.zeros((H, W), dtype='uint8')
            x = int(item[0]) / 490.0 * W
            y = int(item[1]) / 603.0 * H
            angle = float(item[2]) * math.pi / 2
            point1 = (int(x - lenth / 2 * math.cos(angle)), int(y + lenth / 2 * math.sin(angle)))
            point2 = (int(x + lenth / 2 * math.cos(angle)), int(y - lenth / 2 * math.sin(angle)))
            cv2.line(img, point1, point2, 255, 1)
            img = img.astype('float32')
            img = nimg.gaussian_filter(img, sigma)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype('uint8')
            hm_label[i, :, :] = img
        return hm_label

if __name__=="__main__":
    dataset = MultiDataset()
    loader = DataLoader(dataset, batch_size=4)
    for item in loader:
        break
        
        
        



