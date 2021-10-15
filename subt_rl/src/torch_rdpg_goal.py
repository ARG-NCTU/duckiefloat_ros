#! /usr/bin/env python3
import os
import time
import rospy
import torch
import numpy as np
import copy
import random
import math
from collections import deque
from typing import Deque, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sensor_msgs.msg import LaserScan, Joy, Imu
from subt_msgs.srv import pause, start, stop
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

################################## Network #####################################
class Actor(nn.Module):
    def __init__(
        self,
        device,
        in_dim: int,
        out_dim: int,
    ):
        """Initialize."""
        super(Actor, self).__init__()
        self.device = device

        kernel = 3
        stride = 1
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )
        in_dim = 241
        dim = 32 * (in_dim - 2*(kernel-stride))

        self.linear1 = nn.Linear(dim, 512)
        self.linear2 = nn.Linear(512+3, 256)
        self.linear3 = nn.Linear(256, 128)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=False)

        self.out = nn.Linear(128, out_dim)

        init_w = 3e-3
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor,
                hn: torch.Tensor = None,
                cn: torch.Tensor = None) -> torch.Tensor:
        """Forward method implementation."""

        if hn is None:
            hn = torch.zeros((1, 1, 128), device=self.device)
            cn = torch.zeros((1, 1, 128), device=self.device)

        # split to feature, track
        x, other = torch.split(state, 1*241, dim=1)

        # expand to [batch, channel*4, features]
        x = x.reshape(state.shape[0], 1, -1)


        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = torch.cat((x, other), dim=-1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        # reshape to (sequence, batch=1, feature_dim)
        x = x.reshape(x.shape[0], 1, -1)
        x, (hn, cn) = self.lstm(x, (hn, cn))
        # reshape back (batch, feature_dim)
        x = x.reshape(x.shape[0], -1)

        x = self.out(x)

        action = torch.tanh(x)

        return action, hn, cn
################################################################################

class RunRDPGGoal(object):
    def __init__(self):
        super().__init__()
        self.max_dis = 10  # meters
        self.frame = rospy.get_param("~frame", "map")
        self.model = rospy.get_param("~model", "s0312_f2999729.pth")
        self.action_scale = {'linear': 1.5, 'angular': 0.8}

        self.auto = 0
        self.goal = None
        self.pose = None
        self.laser = None
        self.last_pos = None

        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # network
        self.policy_network = Actor(
            self.device, 241, 2).to(self.device)
        my_dir = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(my_dir, "../model/rdpg_torch_goal/" + self.model)
        # model_path = "/home/argsubt/subt-virtual/rl/rdpg/runs/Jul12_19-41-25_yellow-arg7/actor/eval/s0312_f2999729.pth"
        self.policy_network.load_state_dict(torch.load(model_path))

        # lstm hidden state
        self.hn = torch.zeros((1, 1, 128), device=self.device)
        self.cn = torch.zeros((1, 1, 128), device=self.device)


        # pub cmd
        self.pub_cmd = rospy.Publisher("cmd_out", Twist, queue_size=1)

        # subscriber, timer
        self.sub_joy = rospy.Subscriber("joy", Joy, self.cb_joy, queue_size=1)
        self.sub_goal = rospy.Subscriber(
            "goal_in", PoseStamped, self.cb_goal, queue_size=1)
        self.sub_odom = rospy.Subscriber(
            "odom_in", PoseStamped, self.cb_odom, queue_size=1)
        self.sub_laser = rospy.Subscriber(
            "laser_in",  LaserScan, self.cb_laser, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.inference)

    def scale_linear(self, n, bound):
        # bound
        return np.clip(n, 0, 1)*bound

    def scale_angular(self, n, bound):
        return np.clip(n, -1, 1)*bound

    def cb_joy(self, msg):
        start_button = 7
        back_button = 6

        if (msg.buttons[start_button] == 1) and not self.auto:
            self.auto = 1
            rospy.loginfo('go auto')
        elif msg.buttons[back_button] == 1 and self.auto:
            self.auto = 0
            rospy.loginfo('go manual')

    def cb_goal(self, msg):
        if msg.header.frame_id != self.frame:
            self.goal = None
            return

        self.goal = np.array([
            msg.pose.position.x, msg.pose.position.y])

    def cb_odom(self, msg):
        if self.goal is None:
            self.pose = None
            return

        new_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        self.last_pos = new_pos

        # caculate angle diff
        diff = self.goal - self.last_pos[:2]
        r = R.from_quat([msg.pose.orientation.x,
                         msg.pose.orientation.y,
                         msg.pose.orientation.z,
                         msg.pose.orientation.w])
        yaw = r.as_euler('zyx')[0]
        angle = math.atan2(diff[1], diff[0]) - yaw
        if angle >= np.pi:
            angle -= 2*np.pi
        elif angle <= -np.pi:
            angle += 2*np.pi

        diff = np.array(diff)
        track_pos = np.append(diff, angle)
        track_pos = track_pos.reshape(-1)
        self.pose = track_pos

    def cb_laser(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.clip(ranges, 0, self.max_dis)
        self.laser = ranges.reshape(-1)

    def inference(self, event):
        if self.goal is None:
            rospy.loginfo("goal is None")
            return
        if self.pose is None:
            rospy.loginfo("pose is None")
            return
        if self.laser is None:
            rospy.loginfo("laser is None")
            return
        if self.auto == 0:
            rospy.loginfo("is not auto")
            return

        dis = np.linalg.norm(self.goal-self.last_pos)
        if dis < 1.0:
            rospy.loginfo("goal reached")
            self.goal = None
            return


        rospy.loginfo("moving to goal")
        # reshape
        laser = self.laser.reshape(-1)
        track = self.pose.reshape(-1)
        state = np.append(laser, track)

        # feed policy and get action
        state_t = torch.FloatTensor(state).to(self.device)
        state_t = torch.unsqueeze(state_t, dim=0)
        selected_action, self.hn, self.cn = self.policy_network(
            state_t, self.hn, self.cn)
        selected_action = selected_action.detach().cpu().numpy()[0]

        # do action
        action = [0, 0]
        action[0] = self.scale_linear(selected_action[0], self.action_scale['linear'])
        action[1] = self.scale_angular(selected_action[1], self.action_scale['angular'])
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        self.pub_cmd.publish(cmd_vel)

if __name__ == "__main__":
    rospy.init_node("rdpg_goal_nav_rl")
    goalNav = RunRDPGGoal()
    rospy.spin()
