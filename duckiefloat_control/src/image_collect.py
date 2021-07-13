#!/usr/bin/env python

import numpy as np
import rospy
#import picamera
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
import cv2
import time


class Collect(object):
    def __init__(self):
        self.IMG = None
        self.i = 0
        self.img = rospy.Subscriber('img', Image, self.save, queue_size = 1)
    def save(self, data):
        self.IMG = bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")
        cv2.imwrite("/home/ubuntu/subt-duckiefloat/catkin_ws/src/duckiefloat_control/src/Line_data/"+str(self.i)+'.jpg', self.IMG)
        rospy.loginfo(str(self.i))
        self.i += 1


if __name__ == "__main__":
    rospy.init_node("collect")
    collecter = Collect()
    rospy.spin()
