import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import urx
import math3d
import rospy
from options import home_j
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time
#from HandCtrl.CtrlHand import grasp2
from threading import Thread
import threading
import sys
import matplotlib.pyplot as plt
import math
import os
import copy
import random
import cv2


import PIL
import scipy.ndimage as ndimg
import torch

if not os.path.exists('./data'):
    os.mkdir('./data')
if not os.path.exists('./data/images/'):
    os.mkdir('./data/images/')
    os.mkdir('./data/depths/')
    os.mkdir('./data/label_txt/')
class BinEnv():
    
    def __init__(self, image_topic='/kinect2/hd/image_color_rect', depth_topic='/kinect2/hd/image_depth_rect'):

        rospy.init_node('BinPickingEnv')
        self.image_topic = image_topic
        self.depth_topic = depth_topic

        self.ur = urx.Robot('192.168.1.101', use_rt=True)
        self.roi_cor = np.load('roi_cor.npy')

        self.max_yaw = 1.0
        self.min_yaw = -1.0
        self.depth = None
        self.image = None
        self.image_show = None
        self.samples = [(0,0), (0,0)]
        self.location = [0, 0, 0]
        self.label_list = []
        self.start_action = False
        self.get_image_once = False
        self.get_depth_once = False
        self.count_episode = 1344
        self.count_clip = 8240
        self.mouse_click_flip = 1

        # publish var
        self.pub_color_grasp = rospy.Publisher('/color_grasp', Image, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.get_image)
        rospy.Subscriber(self.depth_topic, Image, self.get_depth)

        # visualization thread
        self.vis_thread = Thread(target=self.vis_sample)
        self.vis_thread.daemon = True
        self.vis_thread.start()

        # action thread
        self.check_thread = Thread(target=self.check_action)
        self.check_thread.daemon = True
        self.check_thread.start()

        self.__running = threading.Event()
        time.sleep(2)
        
    def get_image(self, msg):
        if not self.get_image_once:
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(msg)
            roi_cor = self.roi_cor
            img_src = img.copy()[roi_cor[0, 1]:roi_cor[1, 1], roi_cor[0, 0]:roi_cor[1, 0]]
            color_grasp = img_src[:, :int(img_src.shape[1]/2)-45, :]
            self.image = color_grasp
            self.image_show = self.image.copy()
            color_grasp_msg = bridge.cv2_to_imgmsg(color_grasp, encoding='bgr8')
            # publish the state's rgb image
            self.pub_color_grasp.publish(color_grasp_msg)
        self.get_image_once = True
        
    def get_depth(self, msg):
        if not self.get_depth_once:
            bridge = CvBridge()
            depth_src = bridge.imgmsg_to_cv2(msg)
            roi_cor = self.roi_cor
            # depth in uint m
            depth_m = copy.deepcopy(depth_src.astype('float32')/1000.0)
            # get the roi from depth_m
            depth = depth_m[roi_cor[0, 1]:roi_cor[1, 1], roi_cor[0, 0]:roi_cor[1, 0]]
            depth = depth[:, :int(depth.shape[1]/2)-45]

            # define far and near w.r.t camera z axis
            far = 0.73
            near = 0.66

            depth[depth>far] = 0.
            depth[depth<near] = 0.

            depth[depth>0] = (depth[depth>0]- near) / (far -near)
            depth[depth>0] = 255 - depth[depth>0]*255

            # dog filter to sharp the image
            depth_g1 = ndimg.gaussian_filter(depth, 2) 
            depth_g2 = ndimg.gaussian_filter(depth, 20) 

            depth_dog = depth_g1 - depth_g2
            depth_dog[depth_dog < 15] = 0
            depth_dog[depth_dog>0] *= 2
            depth_dog[depth_dog>255] = 255
            
            kernel = np.ones((5, 5), np.uint8)
            depth_dog = cv2.erode(depth_dog, kernel)

            self.depth = depth_dog.astype('uint8')
        self.get_depth_once = True

    def check_action(self):
        while True:
            if self.start_action:
                self.step()
                self.start_action = False

    def vis_sample(self):
        while True:
            if self.get_image_once and self.get_depth_once:
                cv2.putText(self.image_show, 'Press q to exit', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            1)
                cv2.namedWindow('sample')
                cv2.setMouseCallback('sample', self.HIL_callback)
                cv2.imshow('sample', self.image_show)
            key = cv2.waitKey(1)

            # press 'q'
            if key == 113:
                print('closing all !!!')
                self.close()

            # press 'c' to save clip
            if key == 99:
                cv2.line(self.image_show, self.samples[0], self.samples[1], (255, 0, 0), 2)
                right_pos = self.samples[1]
                left_pos = self.samples[0]

                dy = - (right_pos[1] - left_pos[1])
                dx = right_pos[0] - left_pos[0]
                angle = math.atan2(dy, dx) / np.pi * 180
                center_x = int((right_pos[0]+left_pos[0])/2)
                center_y = int((right_pos[1]+left_pos[1])/2)
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle = 180 + angle
                print('angle {}'.format(angle))
                cv2.putText(self.image_show, '%.2f' % (angle), (self.samples[0][0] + 15, self.samples[0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (155, 225, 255), 2)
                self.location = [center_x, center_y, angle/90]
                self.label_list.append(self.location)
                self.save_clip(self.image, self.depth, self.location)
                # yaw from (-1, 1) stands for (-pi/2, pi/2)

            # press s to save episode data
            if key == 115:
                self.save_episode_data()

            # press n to get next episode
            if key == 110:
                self.get_depth_once = False
                self.get_image_once = False
                self.label_list = []

            # press a to execute sample action
            if key == 97:
                self.start_action = True

    def HIL_callback(self, event, x, y, flags, param):
        """
        Human in the loop labeling
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mouse_click_flip:
                self.samples[0] = (x, y)
                print('L click')
                cv2.circle(self.image_show, (x, y), 4, (0, 0, 255), -1)
                self.mouse_click_flip = 1 - self.mouse_click_flip
            else:
                self.samples[1] = (x, y)
                cv2.circle(self.image_show, (x, y), 4, (0, 255, 0), -1)
                print('R click')
                self.mouse_click_flip = 1 - self.mouse_click_flip

    def step_reset(self):
        cur_pos = self.ur.get_pos()
        self.ur.set_pos([cur_pos[0], cur_pos[1], 0.45], vel=0.2, acc=0.1)
        self.ur.movej(home_j, vel=0.4, acc=0.4)
        # avoid gripper block
        time.sleep(1)

    def step_reset2(self):
        cur_pos = self.ur.get_pos()
        self.ur.set_pos([cur_pos[0], cur_pos[1], 0.45], vel=0.2, acc=0.1)
        self.ur.movej(home_j, vel=0.4, acc=0.4)
        # avoid gripper block
        time.sleep(1)

    def episode_reset(self):
        self.ur.movej(home_j, vel=0.4, acc=0.4)
        grasp2(10)
        time.sleep(0.5)
        self.depth = None
        self.image = None
        self.label_list = []
        self.get_image_once = False
        self.get_depth_once = False

    def step(self):
        time.sleep(1)
        x = float(self.location[1]) / float(self.image.shape[0]) * 0.40+0.01 - 0.61
        y = float(self.location[0]) / float(self.image.shape[1]) * (0.73/2-0.01) - 0.27
        angle = self.location[2]
        # augmentation from z, yaw to x, y, z, yaw
        print('step action {}'.format(self.location))
        # translation
        self.ur.set_pos([x, y, 0.4], vel=0.2, acc=0.1)
        # rotation
        self.rotate_tool(angle)
        # fetch
        self.ur.set_pos([x, y, 0.32], vel=0.2, acc=0.1)
        grasp2(110)
        time.sleep(0.5)
        self.ur.translate([0, 0, 0.05],vel=0.2,acc=0.1)
        time.sleep(0.5)
        self.ur.translate([0, 0, -0.05], vel=0.2, acc=0.1)
        grasp2(10)
        # step reset
        self.step_reset()

        
    def force_check(self):
        self.force_set = []
        while True:
            #print('force checking')
            force = self.ur.get_force()
            if force > 70:
                print('force too big')     
                self.step_reset2()
            self.force_set.append(force)
            time.sleep(0.1)
        
    def close(self):
        self.ur.close()
        self.__running.clear()
        os._exit(0)

    def rotate_tool(self, angle, wait=True):
        angle = angle * math.pi/2
        ori = math3d.Orientation()
        ori.rotateZ(angle)
        ori_ref = self.ur.get_orientation()
        ori_new = ori.__mul__(ori_ref)
        self.ur.set_orientation(ori_new, vel=0.2, acc=0.1, wait=wait)

    def save_clip(self,image,depth,location):
        clip_end_x = random.randint(60, 100)
        clip_end_y = random.randint(60, 100)
        clip_size = 160
        cor_x = location[0]
        cor_y = location[1]
        angle = location[2]
        clip_x1 = cor_x - clip_end_x
        clip_x2 = cor_x + clip_size-clip_end_x
        clip_y1 = cor_y - clip_end_y
        clip_y2 = cor_y + clip_size-clip_end_y
        clip_x = np.clip(np.array([clip_x1, clip_x2]), 0, image.shape[1]-1)
        clip_y = np.clip(np.array([clip_y1, clip_y2]), 0, image.shape[0]-1)
        clip_cor_x = cor_x-clip_x[0]
        clip_cor_y = cor_y-clip_y[0]
        image_clip = image[clip_y[0]:clip_y[1], clip_x[0]:clip_x[1], :]
        depth_clip = depth[clip_y[0]:clip_y[1], clip_x[0]:clip_x[1]]
        cv2.imwrite('./data/clip/image_clip/{}_x{}_y{}_{:.3f}_image_clip.png'.format(self.count_clip,clip_cor_x,clip_cor_y,angle),
                    image_clip)
        cv2.imwrite('./data/clip/depth_clip/{}_x{}_y{}_{:.3f}_depth_clip.png'.format(self.count_clip,clip_cor_x,clip_cor_y,angle),
                    depth_clip)
        self.count_clip += 1

    def save_episode_data(self):
        length = len(self.label_list)
        cv2.imwrite('./data/images/{}_rgb.png'.format(self.count_episode), self.image)
        cv2.imwrite('./data/depths/{}_depth.png'.format(self.count_episode), self.depth)
        with open('./data/label_txt/{}.txt'.format(self.count_episode),'w') as f:
            for label_index in range(length):
                label = self.label_list[label_index]
                f.write('{}\t{}\t{:.2f}\n'.format(label[0],label[1],label[2]))
        print('episode{} data has been saved'.format(self.count_episode))
        self.count_episode += 1


if __name__ == '__main__':
    env = BinEnv()
    env.episode_reset()
    while True:
        pass
