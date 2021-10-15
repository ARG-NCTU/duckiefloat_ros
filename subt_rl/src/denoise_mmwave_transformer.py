#!/usr/bin/env python3

import rospy
import os
import time
import copy
import torch
import rospy
import numpy as np
import traceback
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import LaserScan, Joy, Imu
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ti_mmwave_rospkg.msg import RadarScan

from network_transformer import RadarTransformer


class DenoiseMMwave(object):
    def __init__(self):
        # device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # model
        model = rospy.get_param('~model', 'transformer_cgan.pth')
        my_dir = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(
            my_dir, "../model/transformer/" + model)

        # restore network
        self.network = RadarTransformer(
            features=7,
            embed_dim=64,
            nhead=8,
            encoder_layers=6,
            decoder_layers=6,
        ).to(self.device)
        self.network.load_state_dict(torch.load(model_path))
        self.network.eval()

        # publisher
        self.lsr = LaserScan()
        self.lsr.header.frame_id = "base_link"
        self.lsr.range_max = 4.9
        self.lsr.range_min = 0
        self.lsr.angle_max = 2.094395
        self.lsr.angle_min = -2.094395
        self.lsr.angle_increment = 0.017453

        self.pub_laser = rospy.Publisher(
            'recon_laser', LaserScan, queue_size=1)

        # husky radar module transformation
        '''
        unable to import tf in ros melodic python3
        upgrade to ros noetic and use tf listener instead
        '''
        self.module_tf = {
            'quat': [
                [0.000, 0.000, -0.383, 0.924],  # right front
                [0.000, 0.000, 0.383, 0.924],   # left front
                [0.000, 0.000, -0.707, 0.707],  # right back
                [0.000, 0.000, 0.707, 0.707]    # left back
            ],
            'trans': [
                [0.462, -0.186, 0.455],
                [0.462, 0.186, 0.455],
                [-0.086, -0.300, 0.455],
                [-0.086, 0.300, 0.455],
            ]
        }
        self.btf = np.zeros((4, 4, 4))
        for i in range(4):
            self.btf[i] = self.make_tf(
                quaternion=self.module_tf['quat'][i],
                translation=self.module_tf['trans'][i],
            )

        # radar stack
        self.radar_points = []

        # timer for frame
        self.timer = rospy.Timer(rospy.Duration(0.1), self.transform)

        # subscriber radar
        self.sub_mm0 = rospy.Subscriber(
            'radar_0/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm0, queue_size=100)
        self.sub_mm1 = rospy.Subscriber(
            'radar_1/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm1, queue_size=100)
        self.sub_mm2 = rospy.Subscriber(
            'radar_2/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm2, queue_size=100)
        self.sub_mm3 = rospy.Subscriber(
            'radar_3/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm3, queue_size=100)

        rospy.loginfo('node init')

    def make_tf(self, quaternion, translation):
        tf = np.zeros((4, 4))
        quat = R.from_quat(quaternion)
        tf[:3, :3] = quat.as_matrix()
        tf[:3, -1] = translation
        tf[-1, -1] = 1
        return tf

    def cb_mm0(self, msg):
        self.process_radar(msg, 0)

    def cb_mm1(self, msg):
        self.process_radar(msg, 1)

    def cb_mm2(self, msg):
        self.process_radar(msg, 2)

    def cb_mm3(self, msg):
        self.process_radar(msg, 3)

    def process_radar(self, msg, module):
        p = np.array([msg.x, msg.y, msg.z, 1])
        p = np.matmul(self.btf[module], p.T).T[:3]
        self.radar_points.append(
            [rospy.Time.now().to_sec(), module, msg.point_id, p[0], p[1], p[2], msg.velocity])

    def transform(self, e):
        data = np.array(self.radar_points)
        data[:, 0] = rospy.Time.now().to_sec()-data[:, 0]  # relative time

        r_t = torch.Tensor(data).to(self.device)
        r_t = torch.unsqueeze(r_t, dim=1)
        l_t = self.network(r_t, None)
        l_t = l_t.detach().cpu().numpy()
        l_t = np.clip(l_t, 0, 5)
        self.lsr.ranges = l_t.tolist()
        self.lsr.header.stamp = rospy.Time.now()
        self.pub_laser.publish(self.lsr)

        self.radar_points = []


if __name__ == "__main__":
    rospy.init_node('denoise')
    denoise = DenoiseMMwave()
    rospy.spin()
