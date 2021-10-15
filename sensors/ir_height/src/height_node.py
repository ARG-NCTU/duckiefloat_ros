#!/usr/bin/env python
import sys
import time
import os
import rospy
import signal
import Adafruit_ADS1x15
from std_msgs.msg import Empty
from sensor_msgs.msg import Range


class HeightNode(object):
    GAIN = 1
    def __init__(self):
        self.adc = Adafruit_ADS1x15.ADS1115()
        self.adc.start_adc(0, gain=self.GAIN)
        self.distance = 0

        # values used to define the slope and intercept of distance as a function of voltage

        self.m = 1644
        self.p = -0.885

        ### Publisher ###
        self.pub_height = rospy.Publisher("height", Range, queue_size=1)
        
        rospy.loginfo("Initialized")


    def get_height(self):
        voltage = self.adc.get_last_result()
        if voltage <= 0:
            voltage = 1
            rospy.logerr("ERROR: BAD VOLTAGE!!!")
            return
        #rospy.loginfo("voltage = " + str(voltage))
        self.distance = pow(voltage, self.p) * self.m
        #rospy.loginfo("distance = " + str(self.distance))
        msg = Range()
        msg.max_range = 1.5
        msg.min_range = 0
        msg.range = self.distance
        msg.header.stamp = rospy.Time.now()
        self.pub_height.publish(msg)

    def on_shutdown(self):
        rospy.logwarn("Stopping Height Node")

if __name__ == '__main__':
    rospy.init_node('height_node', anonymous=False)

    timer = rospy.timer.Rate(50)
    node = HeightNode()

    rospy.on_shutdown(node.on_shutdown)
    # Keep it spinning to keep the node alive
    while not rospy.is_shutdown():
        node.get_height()
        timer.sleep()
