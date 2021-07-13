#!/usr/bin/env python

import rospy
from motor_hat_driver import MotorHatDriver
from geometry_msgs.msg import Twist
from subt_msgs.srv import stop
from std_msgs.msg import Bool

class ControlNode(object):
    def __init__(self):
        self.motors = MotorHatDriver()
        self.e_stop_on = False

        self.trim = -0.08

        self.sub_cmd_vel = rospy.Subscriber("/cmd_vel", Twist, self.cb_cmd, queue_size = 1)

        self.sub_estop = rospy.Subscriber("/e_stop", Bool, self.cb_e_stop, queue_size=1)
        #self.srv = rospy.Service("/e_stop", stop, self.handle_e_stop)
    def cb_e_stop(self, msg):
        self.e_stop_on = msg.data
        if self.e_stop_on:
            self.motors.setMotorSpeed(0, 0, 0, 0)

    def cb_cmd(self, msg):
        # rospy.loginfo("cmd_cb")

        if self.e_stop_on: 
            rospy.logerr("e_stop is on")
            return

        m1 = -msg.linear.z
        m2 = max(msg.angular.z, 0)
        m3 = msg.linear.x
        m4 = min(msg.angular.z, 0)

        self.motors.setMotorSpeed(m1=m1, m2=m2, m3=m3, m4=m4)
    def on_shutdown(self):
        self.motors.setMotorSpeed(0, 0, 0, 0)
        rospy.logwarn("motors off")


if __name__ == "__main__":
    rospy.init_node("control_node")

    node = ControlNode()
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()
