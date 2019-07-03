from models import Bottleneck, HourGlass
from skimage.io import imread
import numpy as np
import urx
import math3d
import rospy
from options import Options
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from torchvision.transforms import ToTensor
import time
from HandCtrl.CtrlHand import grasp2
from threading import Thread
import threading
import matplotlib.pyplot as plt
import os
import math
import random
import cv2
from numpy.linalg import eig
from post_utils import gcf_extract,vis_gcfs
import torch

opt = Options()

class RealEnv(object):

    def __init__(self, image_topic='/kinect2/hd/image_color_rect', depth_topic='/kinect2/hd/image_depth_rect'):
        rospy.init_node('RealExp')
        self.image_topic = image_topic
        self.depth_topic = depth_topic

        self.ur = urx.Robot('192.168.1.101', use_rt=True)
        self.get_image_once = False
        self.reward = 0
        self.image = None
        self.heatmap = None
        self.heatmap_show = False
        self.hm_overall_show = None
        self.gcf_finish =False
        self.gcf_blocks = None
        self.gcf_cs = None
        self.gcf_rs = None
        self.image_show = None
        self.grasp_done = False

        # publish var
        self.pub_color_grasp = rospy.Publisher('/color_grasp', Image, queue_size=1)

        rospy.Subscriber(self.image_topic, Image, self.get_image)

        self.flip_over = 0

        # # visualization thread
        self.vis_thread = Thread(target=self.visulization)
        self.vis_thread.daemon = True
        self.vis_thread.start()

        self.__running = threading.Event()
        time.sleep(2)

    def get_image(self, msg):
        if not self.get_image_once:
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(msg)
            roi_cor = np.load('roi_cor.npy')
            img_src = img.copy()[roi_cor[0, 1]:roi_cor[1, 1], roi_cor[0, 0]:roi_cor[1, 0]]
            color_grasp = img_src[:, :int(img_src.shape[1] / 2)-45, :]
            self.image = color_grasp

            color_grasp_msg = bridge.cv2_to_imgmsg(color_grasp, encoding='bgr8')
            self.pub_color_grasp.publish(color_grasp_msg)

    def visulization(self):
        while True:
            if self.get_image_once and self.gcf_finish:
                plt.subplot(1,2,1)
                image_show = cv2.resize(self.image_show,(256,256))
                plt.imshow(image_show[...,::-1])
                plt.subplot(1,2,2)
                plt.imshow(self.heatmap)
                plt.pause(0.1)

    def reset1(self):
        self.ur.movej(opt.home_j, vel=0.4, acc=0.4)
        grasp2(160)
        time.sleep(1)

    def reset(self,angle=0.0):
        cur_pos = self.ur.get_pos()
        self.ur.set_pos([cur_pos[0], cur_pos[1], 0.45], vel=0.2, acc=0.1)
        self.ur.set_pos([-0.40612+random.random()*0.05, 0.27505+random.random()*0.05, 0.40129], vel=0.2, acc=0.1)
        self.ur.translate([0, 0, -0.15], vel=0.2, acc=0.2)
        grasp2(160)
        time.sleep(0.5)
        self.ur.translate([0, 0, 0.15], vel=0.2, acc=0.2)
        self.rotate_tool(angle)
        # avoid gripper block
        time.sleep(1)

    def step(self, hm_list):
        gcfs = gcf_extract(hm_list)
        self.gcf_finish = True
        image = self.image.copy()
        self.image_show = vis_gcfs(gcfs,image)
        self.get_image_once = True
        for gcf in gcfs:
            x = gcf[0]
            y = gcf[1]
            angle = gcf[2]
            action_aug = [0.0] * 4
            action_aug[0] = y/600*0.40-0.615
            action_aug[1] = x/1084*0.73-0.35
            action_aug[2] = 0.20
            action_aug[3] = angle/math.pi*180

            print('step action {}'.format(action_aug))
            # translation
            self.ur.set_pos([action_aug[0], action_aug[1], 0.4], vel=0.2, acc=0.1)
            # rotation
            self.rotate_tool(action_aug[3])
            # fetch
            self.ur.set_pos([action_aug[0], action_aug[1], action_aug[2]], vel=0.2, acc=0.1)

            grasp2(60)
            time.sleep(0.5)
            # step reset
            self.reset(-action_aug[3])

    def force_check(self):
        self.force_set = []
        while True:
            # print('force checking')
            force = self.ur.get_force()
            if force > 70:
                print('force too big')
                self.reset()
            self.force_set.append(force)
            time.sleep(0.1)

    def close(self):
        self.ur.close()
        self.__running.clear()
        os._exit(0)

    def rotate_tool(self, angle, wait=True):
        angle = angle/180 * math.pi
        ori = math3d.Orientation()
        ori.rotateZ(angle)
        ori_ref = self.ur.get_orientation()
        ori_new = ori.__mul__(ori_ref)
        self.ur.set_orientation(ori_new, vel=0.2, acc=0.1, wait=wait)


if __name__ == '__main__':
    net = HourGlass(Bottleneck,num_classes=10)
    net.load_state_dict(torch.load('checkpoints/ep_40.pt'))
    net.cpu()
    env = RealEnv()
    env.reset1()
    index = 1
    image = cv2.resize(env.image.copy(), (256,256))
    image = image[..., ::-1]
    input_image = torch.FloatTensor(np.transpose(image[None, :, :, :].astype('float32')/255.0 , [0, 3, 1, 2]))
    output_hm = net(input_image)
    hm = []
    for j in range(output_hm.shape[1]):
        heatmap = output_hm[0, j, :, :].detach().numpy() * 255
        heatmap = np.clip(heatmap, 0, 255)
        hm.append(heatmap)
    hm_overall = sum(hm) / len(hm)
    hm_overall = hm_overall.astype('uint8')
    env.heatmap = hm_overall
    env.heatmap_show = True
    env.step(hm)
    env.close()
