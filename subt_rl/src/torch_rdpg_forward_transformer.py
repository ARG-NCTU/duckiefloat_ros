#! /usr/bin/env python3

import os
import time
import rospy
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import LaserScan, Joy, Imu
from subt_msgs.srv import pause, start, stop
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ti_mmwave_rospkg.msg import RadarScan

from network_forward import Actor, ActorLatent, ActorRadar
from network_transformer import RadarEncoder


class RunRDPGModel(object):
    def __init__(self):
        model = rospy.get_param('~actor_model', 's1536_f1869509.pth')
        model_encoder = rospy.get_param(
            '~encoder_model', 'transformer_encoder_forward.pth')
        self.max_dis = 5
        self.action_scale = {'linear': 0.4, 'angular': 0.35}

        self.auto = 0
        self.estop = False
        self.odom = None
        self.last_pos = None

        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # network
        obs_dim = 243
        action_dim = 2
        my_dir = os.path.abspath(os.path.dirname(__file__))
        actor_model_path = os.path.join(
            my_dir, "../model/rdpg_torch_forward/" + model)
        encoder_model_path = os.path.join(
            my_dir, "../model/transformer/" + model_encoder)

        self.policy_network = ActorRadar(
            self.device,
            obs_dim,
            action_dim,
            actor_state_dict=actor_model_path,
            encoder_state_dict=encoder_model_path,
        ).to(self.device)
        self.policy_network.eval()

        # lstm hidden state
        self.hn = torch.zeros((1, 1, 128), device=self.device)
        self.cn = torch.zeros((1, 1, 128), device=self.device)

        # service
        rospy.Service('RL/pause', pause, self.pause)
        rospy.Service('RL/stop', stop, self.stop_rl)
        rospy.Service('RL/start', start, self.start_rl)

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

        # subscriber
        self.sub_estop = rospy.Subscriber(
            'e_stop', Bool, self.cb_estop, queue_size=1)
        self.sub_odom = rospy.Subscriber(
            'odom', Odometry, self.cb_odom, queue_size=1)
        self.sub_joy = rospy.Subscriber(
            'joy', Joy, self.cb_joy, queue_size=1)

        # publisher
        self.pub_twist = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # timer
        time.sleep(1)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_pub)

        # subscriber radar
        self.sub_mm0 = rospy.Subscriber(
            'radar_0/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm0, queue_size=100)
        self.sub_mm1 = rospy.Subscriber(
            'radar_1/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm1, queue_size=100)
        self.sub_mm2 = rospy.Subscriber(
            'radar_2/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm2, queue_size=100)
        self.sub_mm3 = rospy.Subscriber(
            'radar_3/ti_mmwave/ti_mmwave/radar_scan', RadarScan, self.cb_mm3, queue_size=100)

        rospy.loginfo('press start')

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

    def cb_estop(self, msg):
        self.estop = msg.data

    def cb_odom(self, msg):
        self.odom = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y])
        if self. last_pos is None:
            self.last_pos = self.odom

    def stop_rl(self, req):
        if self.auto == 1:
            response = "stop navigation"
            self.auto = 0
        else:
            response = "joy stick mode"
        rospy.loginfo(response)
        return response

    def start_rl(self, req):
        if self.auto == 0:
            response = "start navigation"
            self.hn = torch.zeros((1, 1, 128), device=self.device)
            self.cn = torch.zeros((1, 1, 128), device=self.device)
            self.auto = 1
        else:
            response = "running"
        rospy.loginfo(response)
        return response

    def pause(self, req):
        if self.auto == 1:
            response = "pause for 10 second"
            self.pause_timer = rospy.Timer(
                rospy.Duration(10), self.resume_navigation)
            self.auto = 0
        else:
            response = "joy stick mode"
        rospy.loginfo(response)
        return response

    def timer_pub(self, event):
        if len(self.radar_points) != 0 \
                and self.auto != 0 \
                and self.estop != True \
                and self.odom is not None:

            # prepare state
            data = np.array(self.radar_points)
            data[:, 0] = rospy.Time.now().to_sec()-data[:, 0]  # relative time
            r_t = torch.Tensor(data).to(self.device)
            r_t = torch.unsqueeze(r_t, dim=1)
            pos_diff = torch.Tensor(self.odom - self.last_pos).to(self.device)
            self.last_pos = self.odom

            # feed policy
            action, self.hn, self.cn = self.policy_network(
                r_t, pos_diff, self.hn, self.cn)
            action = action.detach().cpu().numpy()[0]

            cmd_vel = Twist()
            cmd_vel.linear.x = np.clip(
                action[0], 0, 1)*self.action_scale['linear']
            cmd_vel.angular.z = np.clip(
                action[1], -1, 1)*self.action_scale['angular']
            self.pub_twist.publish(cmd_vel)

            self.radar_points = []

    def on_shutdown(self):
        pass

    def cb_joy(self, joy_msg):
        start_button = 7
        back_button = 6

        if (joy_msg.buttons[start_button] == 1) and not self.auto:
            self.auto = 1
            self.hn = torch.zeros((1, 1, 128), device=self.device)
            self.cn = torch.zeros((1, 1, 128), device=self.device)
            rospy.loginfo('go auto')
        elif joy_msg.buttons[back_button] == 1 and self.auto:
            self.auto = 0
            rospy.loginfo('go manual')


if __name__ == '__main__':
    rospy.init_node('rl_rdpg')
    runmodel = RunRDPGModel()
    rospy.on_shutdown(runmodel.on_shutdown)
    rospy.spin()
