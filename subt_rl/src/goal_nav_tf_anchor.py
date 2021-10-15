#! /usr/bin/env python3
import os
import rospy
import tensorflow as tf
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Joy
from std_msgs.msg import Bool
from scipy.spatial.transform import Rotation as R
from anchor_measure.msg import PoseDirectional

class GoalNav(object):
    def __init__(self):
        super().__init__()
        self.max_dis = 10  # meters
        self.laser_n = 4
        self.pos_n = 10
        self.frame = rospy.get_param("~frame", "map")
        self.action_scale = {'linear': rospy.get_param(
            '~linear_scale', 0.3), 'angular': rospy.get_param("~angular_scale", 0.18)}

        self.auto = 0
        self.goal = None
        self.pos_track = None
        self.laser_stack = None
        self.last_pos = None

        self.last_omega = 0
        self.omega_gamma = 0.25

        self.vel_ratio = 0

        # network
        obs_dim = 243
        action_dim = 2
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)
        my_dir = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(my_dir, "../model/goal/policy")
        self.policy_network = tf.saved_model.load(model_path)

        # pub cmd
        self.pub_cmd = rospy.Publisher("cmd_out", Twist, queue_size=1)

        # subscriber, timer
        self.sub_joy = rospy.Subscriber("joy", Joy, self.cb_joy, queue_size=1)
        self.sub_goal = rospy.Subscriber(
            "goal_in", PoseStamped, self.cb_goal, queue_size=1)
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
        
        print(msg)
        new_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        diff = new_goal
        angle = math.atan2(diff[1], diff[0])
        if angle >= np.pi:
            angle -= 2*np.pi
        elif angle <= -np.pi:
            angle += 2*np.pi

        track_pos = np.append(diff, angle)
        if self.pos_track is None:
            self.pos_track = np.tile(track_pos, (self.pos_n, 1))
        else:
            self.pos_track[:-1] = self.pos_track[1:]
            self.pos_track[-1] = track_pos
        self.goal = new_goal

    def cb_laser(self, msg):
        ranges = np.array(msg.ranges)
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

        dis = np.linalg.norm(self.goal)
        if dis < 0.8:
            rospy.loginfo("goal reached")
            self.goal = None
            return

        # self.vel_ratio = rospy.get_param("/velocity_mode", 4) * (1./5)

        # reshape
        laser = self.laser_stack.reshape(-1)
        track = self.pos_track.reshape(-1)
        state = np.append(laser, track)

        state = tf.convert_to_tensor([state], dtype=tf.float32)

        action = self.policy_network(state)[0].numpy()
        self.last_omega = self.omega_gamma * \
            action[1] + (1-self.omega_gamma)*self.last_omega

        cmd = Twist()
        cmd.linear.x = action[0]*self.action_scale['linear']
        cmd.angular.z = self.last_omega * \
            self.action_scale['angular']

        self.pub_cmd.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("goal_nav_rl")
    goalNav = GoalNav()
    rospy.spin()

