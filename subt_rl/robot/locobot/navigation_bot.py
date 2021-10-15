#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from pyrobot import Robot

import copy
import os
import sys
import termios
import time
import tty

import numpy as np
import yaml

# import signal

from std_msgs.msg import Empty, Bool

class Navi(object):
    def __init__(self):
        
        # Load config file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        root_path = os.path.dirname(os.path.dirname(dir_path))
        cfg_path = os.path.join(root_path, "config", cfg_file)
        with open(cfg_path, "r") as f:
            self.cfg = yaml.load(f)
        
        self.use_arm = rospy.get_param("use_arm")
        self.use_camera = rospy.get_param("use_camera")
        self.use_base = rospy.get_param("use_base")
        self.use_gripper = self.use_arm

        arm_config = {"control_mode": "position", "use_moveit": False}

        self.bot = Robot(
            "locobot",
            use_arm=self.use_arm,
            use_base=self.use_base,
            use_camera=self.use_camera,
            use_gripper=self.use_gripper,
            arm_config=arm_config,
        )

        posn = np.asarray(['0','0','0.785'], dtype=np.float64, order="C")
        self.bot.base.go_to_relative(
        posn, use_map=False, close_loop=True, smooth=True)

        # subscriber, timer
        self.sub_vel_cmd = rospy.Subscriber('vel_cmd', Twist, self.vel_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1), self.inference)

        # Publishers
        self.pub_stop = rospy.Publisher(
            self.cfg["ROSTOPIC_STOP_EXECUTION"], Empty, queue_size=1
        )
        self.base_cmd_pub = rospy.Publisher(
            self.cfg["ROSTOPIC_BASE_COMMAND"], Twist, queue_size=1
        )
        self.safety_status_pub = rospy.Publisher(
            self.cfg["ROSTOPIC_SAFETY_STATUS"], Bool, queue_size=1
        )

        # Initialize arm, gripper and pan-tilt camera
        if self.use_arm:
            # set arm to a good pose
            # if the arm starts from the rest pose,
            # some teleops might break the arm
            self.bot.arm.set_joint_positions(
                np.array([0, -0.14, 0.76, 1.07, 0]), plan=False
            )
        # NOTE - Do not remove the following steps!
        if self.use_gripper:
            self.bot.gripper.reset()

        if self.use_camera:
            self.bot.camera.set_pan(self.cfg["START_PAN"])
            self.bot.camera.set_tilt(self.cfg["START_TILT"])

        self.base_cmd = None
        self.base_cmd_prev = None


        self.status = True
        self.is_robot_moving = False
        self.base_max_vel = 0.7
        self.base_max_ang_rate = 0.95

        self.start_time = time.time()

        self.target_alpha = 0.0
        self.trans, _, _ = self.bot.arm.pose_ee

        rospy.loginfo("LoCoBot is ready now...")
    
    @property
    def pose_fingertip(self):
        """ pose of finger tip """
        return self.bot.arm.get_transform("/base_link", "/finger_l")

    def set_joint(self, joint_target):
        self.is_robot_moving = True
        self.start_time = time.time()
        self.bot.arm.set_joint_positions(joint_target, plan=False, wait=False)



    def check_safety(self, key):
        """
        Simple heuristics for avoiding collisions
        """

        # check the shoulder angle
        if self.bot.arm.get_joint_angle("joint_1") < -0.6:
            rospy.logerr(
                "Possible risk of collision with camera mount when moving back"
            )
            self.safety_status_pub.publish(False)
            return False

        # check if end effector is too close to the floor
        if (
            self.pose_fingertip[0][-1] < self.cfg["COLLISION_Z_THRESHOLD"]
            and not key == self.cfg["KEY_POS_Z"]
        ):
            rospy.logerr("Possible risk of collision with floor")
            self.safety_status_pub.publish(False)
            return False

        self.safety_status_pub.publish(True)
        return True


    def move_base(self, lin_speed, ang_speed):
        if self.use_base:
            if np.fabs(lin_speed) > 0.01 or np.fabs(ang_speed) > 0.01:
                self.is_robot_moving = True
                self.bot.base.set_vel(
                    fwd_speed=lin_speed, turn_speed=ang_speed, exe_time=0.5
                )
            else:
                self.bot.base.stop()

    def reset(self):
        self.set_joint(np.zeros(5))

    def exit(self):
        rospy.loginfo("Exiting...")
        sys.exit(0)

    def signal_handler(self, sig, frame):
        self.exit()
 
    def vel_cb(self, vel_msg):        
        self.base_cmd = vel_msg

    def inference(self, event):
        base_cmd = copy.deepcopy(self.base_cmd)
        linear_vel = max(base_cmd.linear.x, self.base_max_vel)
        ang_rate = max(base_cmd.angular.z, self.base_max_ang_rate)
        arm_cmd = copy.deepcopy(self.arm_cmd)
        arm_cmd = max(arm_cmd, self.arm)

        self.move_base(linear_vel, ang_rate)

        self.base_cmd_prev  = copy.deepcopy(base_cmd)
        self.base_cmd = None

if __name__ == "__main__":

    rospy.init_node("navigation_rl")
    navi = Navi()
    rospy.spin()
