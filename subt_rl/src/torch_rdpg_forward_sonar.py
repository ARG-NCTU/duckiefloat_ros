#! /usr/bin/env python3

import os
import time
import rospy
import torch
import numpy as np

from sensor_msgs.msg import LaserScan, Joy, Imu
from subt_msgs.srv import pause, start, stop
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from network_forward import Actor


class RunRDPGModel(object):
    def __init__(self):
        model = rospy.get_param('~model', 's1536_f1869509.pth')
        self.max_dis = 3
        self.action_scale = {'linear': 0.4, 'angular': 0.3}

        self.auto = 0
        self.scan = None
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
        self.policy_network = Actor(
            self.device, obs_dim, action_dim).to(self.device)
        my_dir = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(
            my_dir, "../model/rdpg_torch_forward/" + model)
        self.policy_network.load_state_dict(torch.load(model_path))

        # lstm hidden state
        self.hn = torch.zeros((1, 1, 128), device=self.device)
        self.cn = torch.zeros((1, 1, 128), device=self.device)

        # service
        rospy.Service('RL/pause', pause, self.pause)
        rospy.Service('RL/stop', stop, self.stop_rl)
        rospy.Service('RL/start', start, self.start_rl)

        # subscriber
        self.sub_estop = rospy.Subscriber(
            'e_stop', Bool, self.cb_estop, queue_size=1)
        self.sub_laser = rospy.Subscriber(
            'laser', LaserScan, self.cb_sc, queue_size=1)
        self.sub_odom = rospy.Subscriber(
            'odom', Odometry, self.cb_odom, queue_size=1)
        self.sub_joy = rospy.Subscriber(
            'joy', Joy, self.cb_joy, queue_size=1)

        # publisher
        self.pub_twist = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # timer
        time.sleep(1)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_pub)
        rospy.loginfo('press start')

    def cb_estop(self, msg):
        self.estop = msg.data

    def cb_sc(self, msg):
        a = np.array(msg.ranges)[:-1]
        a = a.reshape((16, -1))
        a = np.min(a, axis=1)
        print(a)
        na = np.ones((16, 15))*self.max_dis
        for i in range(15):
            na[:, i] = a
        na = np.append(na.reshape(-1), self.max_dis)
        self.scan = np.clip(na, 0, self.max_dis)
        

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
        if self.scan is not None \
                and self.auto != 0 \
                and self.estop != True \
                and self.odom is not None:

            # prepare state
            pos_diff = self.odom - self.last_pos
            state = np.append(self.scan, pos_diff)
            self.last_pos = self.odom
            state_t = torch.FloatTensor(state).to(self.device)
            state_t = torch.unsqueeze(state_t, dim=0)

            # feed policy
            action, self.hn, self.cn = self.policy_network(
                state_t, self.hn, self.cn)
            action = action.detach().cpu().numpy()[0]

            cmd_vel = Twist()
            cmd_vel.linear.x = np.clip(
                action[0], 0, 1)*self.action_scale['linear']
            cmd_vel.angular.z = np.clip(
                action[1], -1, 1)*self.action_scale['angular']
            self.pub_twist.publish(cmd_vel)

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
