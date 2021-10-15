#! /usr/bin/env python3

from network_transformer import RadarEncoder
from network_forward import Actor, ActorLatent, ActorRadar, ActorCNN
from ti_mmwave_rospkg.msg import RadarScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool
from subt_msgs.srv import pause, start, stop
from sensor_msgs.msg import LaserScan, Joy, Imu
import os
import time
import rospy
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class ExtractEmbedding(object):
    def __init__(self):
        self.max_dis = 5
        self.lidar_laser = None
        self.radar_laser = None

        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # network
        obs_dim = 243
        action_dim = 2
        my_dir = os.path.abspath(os.path.dirname(__file__))
        lidar_cnn_path = os.path.join(
            my_dir, "../model/rdpg_torch_forward/s1536_f1869509.pth")
        radar_cnn_path = os.path.join(
            my_dir, "../model/rdpg_torch_forward/mse.pth")
        radar_transformer_path = os.path.join(
            my_dir, "../model/transformer/transformer_encoder_forward.pth")

        self.encoder_transformer = RadarEncoder().to(self.device)
        self.encoder_transformer.load_state_dict(
            torch.load(radar_transformer_path))
        self.encoder_transformer.eval()

        self.encoder_cnn = ActorCNN(
            self.device, obs_dim, action_dim
        ).to(self.device)
        self.encoder_cnn.load_state_dict(torch.load(radar_cnn_path))

        self.encoder = ActorCNN(
            self.device,  obs_dim, action_dim
        ).to(self.device)
        self.encoder.load_state_dict(torch.load(lidar_cnn_path))

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

        self.writer = SummaryWriter('runs/vis_embedding')
        self.embds = []
        self.labels = []
        self.imgs = []
        self.step = 0

        # timer
        time.sleep(1)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_pub)

        # subscriber radar
        self.sub_mm0 = rospy.Subscriber(
            'husky1/radar_0/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm0, queue_size=100)
        self.sub_mm1 = rospy.Subscriber(
            'husky1/radar_1/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm1, queue_size=100)
        self.sub_mm2 = rospy.Subscriber(
            'husky1/radar_2/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm2, queue_size=100)
        self.sub_mm3 = rospy.Subscriber(
            'husky1/radar_3/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm3, queue_size=100)
        self.sub_lidar_laser = rospy.Subscriber(
            'husky1/RL/scan', LaserScan, self.cb_lidar_laser, queue_size=1
        )
        self.sub_radar_laser = rospy.Subscriber(
            'husky1/RL/scan_mmwave', LaserScan, self.cb_radar_laser, queue_size=1)

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

    def cb_lidar_laser(self, msg):
        self.lidar_laser = np.array(msg.ranges)
        self.lidar_laser = np.clip(self.lidar_laser, 0, self.max_dis)

    def cb_radar_laser(self, msg):
        self.radar_laser = np.array(msg.ranges)
        self.radar_laser = np.clip(self.radar_laser, 0, self.max_dis)

    def timer_pub(self, event):
        if len(self.radar_points) != 0 \
                and self.lidar_laser is not None \
                and self.radar_laser is not None:

            data = np.array(self.radar_points)
            data[:, 0] = rospy.Time.now().to_sec()-data[:, 0]  # relative time
            r_t = torch.Tensor(data).to(self.device)
            r_t = torch.unsqueeze(r_t, dim=1)

            l = torch.Tensor(self.radar_laser).to(self.device)
            l = torch.unsqueeze(l, dim=0)

            rl = torch.Tensor(self.lidar_laser).to(self.device)
            rl = torch.unsqueeze(rl, dim=0)

            r_embd = self.encoder_transformer(r_t, None).detach().cpu()
            rl_embd = self.encoder_cnn(rl).detach().cpu()[0]
            l_embd = self.encoder(l).detach().cpu()[0]

            print(r_embd.shape)
            print(rl_embd.shape)
            print(l_embd.shape)
            print('---------------------')

            self.embds.append(r_embd)
            self.embds.append(rl_embd)
            self.embds.append(l_embd)
            self.labels.append('radar_point_%d' % (self.step))
            self.labels.append('radar_scan_%d' % (self.step))
            self.labels.append('lidar_scan_%d' % (self.step))

            tune = max(1,  (50+3*(self.step))/255.)
            zs = torch.ones((1, 1, 40, 40)) - tune
            cs = torch.ones((1, 1, 40, 40))
            label_r = torch.cat((cs, zs, zs), dim=1)
            label_g = torch.cat((zs, cs, zs), dim=1)
            label_b = torch.cat((zs, zs, cs), dim=1)
            self.imgs.append(label_r)
            self.imgs.append(label_g)
            self.imgs.append(label_b)

            self.step += 1
            self.radar_points = []

    def on_shutdown(self):
        wembds = torch.stack(self.embds)
        wimgs = torch.cat(self.imgs)
        print(wembds.shape)
        print(wimgs.shape)
        print(len(self.labels))

        self.writer.add_embedding(
            wembds,
            metadata=self.labels,
            label_img=wimgs
        )

        rospy.loginfo('tensorboard saved')


if __name__ == '__main__':
    rospy.init_node('visualize_embedding')
    runmodel = ExtractEmbedding()
    rospy.on_shutdown(runmodel.on_shutdown)
    rospy.spin()
