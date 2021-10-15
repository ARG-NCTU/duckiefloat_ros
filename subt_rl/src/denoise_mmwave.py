#!/usr/bin/env python3

import rospy
import os
import time
import copy
import torch
import rospy
import numpy as np

from sensor_msgs.msg import LaserScan, Joy, Imu
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from network_vae import MMvae
from network_generator import Generator

class DenoiseMMwave(object):
    def __init__(self):
        # device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.method = rospy.get_param('~method', 'cgan')
        # model
        model = rospy.get_param('~model', '0827_1851.pth')
        my_dir = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(my_dir, "../model/"+ self.method +"/" + model)

        # restore network
        if self.method == "cgan":
            self.network = Generator().to(self.device)
        elif self.method == "vae":
            self.network = MMvae().to(self.device)
            
        self.network.load_state_dict(torch.load(model_path))

        # publisher
        self.pub_laser = rospy.Publisher(
            'recon_laser', LaserScan, queue_size=1)

        # subscriber
        self.sub_mm_laser = rospy.Subscriber(
            'mm_laser', LaserScan, self.cb_mm_laser, queue_size=1)

    def cb_mm_laser(self, msg):
        laser = torch.FloatTensor(msg.ranges)
        laser = laser.reshape(1, 1, -1).to(self.device)

        if self.method == 'vae':
            recon, _, _ = self.network(laser)
        elif self.method == 'cgan':
            recon = self.network(laser)
        recon = recon.detach().cpu().numpy()[0]
        recon = np.clip(recon, 0, 5).astype(np.float32).tolist()[0]

        msg.range_max = 4.9
        msg.ranges = recon

        self.pub_laser.publish(msg)


if __name__ == "__main__":
    rospy.init_node('denoise')
    denoise = DenoiseMMwave()
    rospy.spin()
