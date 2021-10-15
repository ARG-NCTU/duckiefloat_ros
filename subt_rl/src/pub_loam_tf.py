#! /usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf


class SLAMtf(object):
    def __init__(self):
        self.listener = tf.TransformListener()
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.pub_pose = rospy.Publisher(
            "slam_pose", PoseStamped, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.pub_pose_cb)

    def pub_pose_cb(self, e):
        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.Time.now()

        try:
            (trans, rot) = self.listener.lookupTransform(
                self.map_frame, self.robot_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        msg.pose.position.x = trans[0]
        msg.pose.position.y = trans[1]
        msg.pose.position.z = trans[2]
        msg.pose.orientation.x = rot[0]
        msg.pose.orientation.y = rot[1]
        msg.pose.orientation.z = rot[2]
        msg.pose.orientation.w = rot[3]
        self.pub_pose.publish(msg)


if __name__ == "__main__":
    rospy.init_node("SLAMtf")
    slam_tf = SLAMtf()
    rospy.spin()
