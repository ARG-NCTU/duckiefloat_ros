#!/usr/bin/env python
import sys
import time
import os
import rospy
import signal
import Adafruit_ADS1x15
from std_msgs.msg import Float32


class CurrentNode(object):
    GAIN = 1
    def __init__(self):
        self.adc = Adafruit_ADS1x15.ADS1115()
        self.adc.start_adc(2, gain=self.GAIN)
        self.battery_current = 0

        # values used to define the slope and intercept of distance as a function of voltage

        self.m = 5
        self.p = 1

        ### Publisher ###
        self.pub_current = rospy.Publisher("battery_current", Float32, queue_size=1)
        
        rospy.loginfo("Initialized")


    def get_current(self):
        voltage = self.adc.get_last_result()
        if voltage <= 0:
            voltage = 1
            rospy.logerr("ERROR: BAD VOLTAGE!!!")
            return
        #rospy.loginfo("current = " + str(voltage))
        self.battery_current = pow(voltage, self.p) * self.m
        rospy.loginfo("current = " + str(self.battery_current))

        msg = Float32()
        msg = self.battery_current
        #msg.header.stamp = rospy.Time.now()
        self.pub_current.publish(msg)

    def on_shutdown(self):
        rospy.logwarn("Stopping Current Node")

if __name__ == '__main__':
    rospy.init_node('current_node', anonymous=False)

    timer = rospy.timer.Rate(30)
    node = CurrentNode()

    rospy.on_shutdown(node.on_shutdown)
    # Keep it spinning to keep the node alive
    while not rospy.is_shutdown():
        node.get_current()
        timer.sleep()
