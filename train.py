from multigrasp_loader import MultiDataset 
from models import HourGlass, Bottleneck
from torch.utils.data import Dataset,DataLoader
from torch.nn import MSELoss, SmoothL1Loss
from threading import Thread
import matplotlib.pyplot as plt
import threading
import pdb
import torch.optim as optim
import numpy as np
import torch
import cv2
from options import Options

opt = Options()

if __name__ == '__main__':
    bs = 32
    dataset = MultiDataset()
    dataLoader = DataLoader(dataset=dataset,batch_size=bs,shuffle=True)

    # model
    net = HourGlass(Bottleneck, num_classes=10)
    net = net.cuda()

    # loss 
    L2Loss = MSELoss(size_average=True)
    L1Loss = SmoothL1Loss(size_average=True)

    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
    print_freq = 10 
    vis_freq = 20
    for ep in range(opt.epoch):
        ep_loss = 0
        ep_loss_L2 = 0
        ep_loss_L1 = 0
        for i, (x, y) in enumerate(dataLoader):
            y = y.cuda()
            x = x.cuda()
            y_hats = net(x)

            loss1 = 0 * L2Loss(y_hats, y)
            loss2 = 1 * L1Loss(y_hats, y)
            loss = loss1 + loss2

            ep_loss += loss
            ep_loss_L2 += loss1
            ep_loss_L1 += loss2

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        ep_loss /= (len(dataset)/bs)
        print("epoch:{}, MSELoss:{}, L1Loss:{}".format(ep, ep_loss_L2, ep_loss_L1))
        if ep % 5 == 0 :
            torch.save(net.state_dict(), 'checkpoints/ep_{}.pt'.format(ep))
