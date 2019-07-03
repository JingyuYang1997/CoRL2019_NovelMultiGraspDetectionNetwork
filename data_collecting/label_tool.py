import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time
from threading import Thread
import threading
import math
import os
import cv2


if not os.path.exists('./data'):
    os.mkdir('./data')
if not os.path.exists('./data/images/'):
    os.mkdir('./data/images/')
    os.mkdir('./data/label_txt/')


class Label(object):
    def __init__(self, image_topic='/kinect2/hd/image_color_rect'):

        rospy.init_node('Label')
        self.image_topic = image_topic

        self.roi_cor = np.load('roi_cor.npy')
        self.image = None
        self.image_show = None
        self.samples = [(0,0), (0,0)]
        self.location = [0, 0, 0]
        self.label_list = []
        self.start_action = False
        self.get_image_once = False
        self.count_episode = 1
        self.mouse_click_flip = 1

        # publish var
        self.pub_color_grasp = rospy.Publisher('/color_grasp', Image, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.get_image)

        # visualization thread
        self.vis_thread = Thread(target=self.vis_sample)
        self.vis_thread.daemon = True
        self.vis_thread.start()

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

    def vis_sample(self):
        while True:
            if self.get_image_once :
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

            # press 'c' to calculate angle
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

            # press s to save data
            if key == 115:
                self.save_episode_data()

            # press n to get next image
            if key == 110:
                self.get_image_once = False
                self.label_list = []

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

    def reset(self):
        self.image = None
        self.label_list = []
        self.get_image_once = False
        
    def close(self):
        self.__running.clear()
        os._exit(0)

    def save_episode_data(self):
        length = len(self.label_list)
        cv2.imwrite('./data/images/{}_rgb.png'.format(self.count_episode), self.image)
        with open('./data/label_txt/{}.txt'.format(self.count_episode),'w') as f:
            for label_index in range(length):
                label = self.label_list[label_index]
                f.write('{}\t{}\t{:.2f}\n'.format(label[0],label[1],label[2]))
        print('episode{} data has been saved'.format(self.count_episode))
        self.count_episode += 1


if __name__ == '__main__':
    env = Label()
    env.reset()
    while True:
        pass
