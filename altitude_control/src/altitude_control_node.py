#!/usr/bin/env python

import rospy
from simple_pid import PID
from sensor_msgs.msg import Range
from std_msgs.msg import Float32
import numpy as np

class Height_Controller(object):
    def __init__(self):
        rospy.loginfo(rospy.get_name() + ": Initializing")

        self.target_height = 0.5
        self.height = self.target_height
        self.memory = [self.target_height, self.target_height, self.target_height, self.target_height]

        self.sample_time = 0.1
        self.controller = PID(1.4, 0.05, 0.8, sample_time=self.sample_time, setpoint=self.target_height, output_limits=(-1, 1))
        self.sub_height = rospy.Subscriber("height", Range, self.height_cb, queue_size=1)
        self.sub_target = rospy.Subscriber("target_height", Range, self.target_cb, queue_size=1)
        self.pub_control = rospy.Publisher("~control", Float32, queue_size=1)


        rospy.Timer(rospy.Duration(self.sample_time), self.control)
        rospy.loginfo(rospy.get_name() + ": Done Initializing")

    def control(self, event):
        msg = Float32()
        msg.data = self.controller(self.height)
        #rospy.loginfo("cmd = " + str(msg.data))
        self.pub_control.publish(msg)


    def height_cb(self, msg):
        self.memory.append(msg.range)
        self.memory.pop(0)
        #print self.memory
        self.height = np.average(self.memory)

    def target_cb(self, msg):
        pass

if __name__ == "__main__":
    rospy.init_node("altitude_control_node")
    controller = Height_Controller()
    rospy.spin()
    

