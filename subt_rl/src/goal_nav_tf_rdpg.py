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

class RunRDPGGoal(object):
    def __init__(self):
        super().__init__()
        self.max_dis = 10  # meters
        self.frame = rospy.get_param("~frame", "map")
        self.action_scale = {'linear': rospy.get_param('~linear_scale', 0.3), 'angular': rospy.get_param("~angular_scale", 0.18)}

        self.auto = 0
        self.goal = None
        self.pose = None
        self.laser = None
        self.last_pos = None

        self.last_omega = 0
        self.omega_gamma = 0.25

        # network
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
        print('cb goal')
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
        if dis < 0.8:
            rospy.loginfo("goal reached")
            self.goal = None
            return

        # reshape
        laser = self.laser.reshape(-1)
        track = self.pose.reshape(-1)
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
    rospy.init_node("RunRDPGGoal")
    goalNav = RunRDPGGoal()
    rospy.spin()
