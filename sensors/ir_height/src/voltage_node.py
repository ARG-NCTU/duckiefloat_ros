#!/usr/bin/env python
import sys
import time
import os
import rospy
import signal
import Adafruit_ADS1x15
from std_msgs.msg import Float32


class VoltageNode(object):
    GAIN = 1
    def __init__(self):
        self.adc = Adafruit_ADS1x15.ADS1115()
        self.adc.start_adc(1, gain=self.GAIN)
        self.battery_voltage = 0

        # values used to define the slope and intercept of distance as a function of voltage

        self.m = 5
        self.p = 1

        ### Publisher ###
        self.pub_voltage = rospy.Publisher("battery_voltage", Float32, queue_size=1)
        
        rospy.loginfo("Initialized")


    def get_voltage(self):
        voltage = self.adc.get_last_result()
        if voltage <= 0:
            voltage = 1
            rospy.logerr("ERROR: BAD VOLTAGE!!!")
            return
        #rospy.loginfo("voltage = " + str(voltage))
        self.battery_voltage = pow(voltage, self.p) * self.m
        rospy.loginfo("voltage = " + str(self.battery_voltage))

        msg = Float32()
        msg = self.battery_voltage
        #msg.header.stamp = rospy.Time.now()
        self.pub_voltage.publish(msg)

    def on_shutdown(self):
        rospy.logwarn("Stopping Height Node")

if __name__ == '__main__':
    rospy.init_node('voltage_node', anonymous=False)

    timer = rospy.timer.Rate(1)
    node = VoltageNode()

    rospy.on_shutdown(node.on_shutdown)
    # Keep it spinning to keep the node alive
    while not rospy.is_shutdown():
        node.get_voltage()
        timer.sleep()
