#! /usr/bin/env python3
import os
import rospy
import torch
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Joy
from scipy.spatial.transform import Rotation as R
from network_goal import Actor


class GoalNav(object):
    def __init__(self):
        super().__init__()
        self.max_dis = 10  # meters
        self.laser_n = 4
        self.pos_n = 10
        self.action_scale = {'linear': 0.3, 'angular': 0.18}
        self.frame = rospy.get_param("~frame", "map")
        model = rospy.get_param("~model", "s0214_f435052.pth")

        self.auto = 0
        self.goal = None
        self.pos_track = None
        self.laser_stack = None
        self.last_pos = None

        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # network
        obs_dim = 243
        action_dim = 2
        self.policy_network = Actor(obs_dim, action_dim).to(self.device)
        my_dir = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(
            my_dir, "../model/goal/" + model)
        self.policy_network.load_state_dict(torch.load(model_path))

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

    def scale_pose(self, value):
        if value > 0:
            return math.log(1 + value)
        elif value < 0:
            return -math.log(1 + abs(value))

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
            self.pos_track = None
            return

        # caculate angle diff
        new_pos = np.array(
            [msg.pose.position.x, msg.pose.position.y])
        diff = self.goal - new_pos
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

        # update pose tracker
        diff = np.array([self.scale_pose(v) for v in diff])
        track_pos = np.append(diff, angle)
        if self.pos_track is None:
            self.pos_track = np.tile(track_pos, (self.pos_n, 1))
        else:
            self.pos_track[:-1] = self.pos_track[1:]
            self.pos_track[-1] = track_pos
        self.last_pos = new_pos

    def cb_laser(self, msg):
        ranges = np.array(msg.ranges)
        min_r = np.min(ranges)
        if min_r <= 0.4:
            rospy.logwarn('danger')
            self.auto = 0

        ranges = np.clip(ranges, 0, self.max_dis)
        if self.laser_stack is None:
            self.laser_stack = np.tile(ranges, (self.laser_n, 1))
        else:
            self.laser_stack[:-1] = self.laser_stack[1:]
            self.laser_stack[-1] = ranges


    def inference(self, event):
        if self.goal is None:
            return
        if self.pos_track is None:
            return
        if self.laser_stack is None:
            return
        if self.auto == 0:
            return

        dis = np.linalg.norm(self.last_pos-self.goal)
        if dis < 1:
            rospy.loginfo("goal reached")
            self.goal = None
            return

        # reshape
        laser = self.laser_stack.reshape(-1)
        track = self.pos_track.reshape(-1)
        state = np.append(laser, track)

        state = torch.FloatTensor(state).to(self.device)
        state = torch.unsqueeze(state, dim=0)

        action = self.policy_network(state).detach().cpu().numpy()[0]

        cmd = Twist()
        cmd.linear.x = action[0]*self.action_scale['linear']
        cmd.angular.z = action[1]*self.action_scale['angular']

        self.pub_cmd.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("goal_nav_rl")
    goalNav = GoalNav()
    rospy.spin()
