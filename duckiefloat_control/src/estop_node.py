#!/usr/bin/env python

import rospy
from subt_msgs.srv import stop
from std_msgs.msg import Bool

class E_stop(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " % (self.node_name))

        self.xbee_stopped = False
        self.lora_stopped = False
        self.joy_stopped = False
        self.e_stop_on = False
        self.release = False

        self.sub_xbee = rospy.Subscriber("/e_stop_xbee", Bool, self.cb_xbee, queue_size=1)
        self.sub_lora = rospy.Subscriber("/e_stop_lora", Bool, self.cb_lora, queue_size=1)
        self.sub_joy = rospy.Subscriber("/e_stop_joy", Bool, self.cb_joy, queue_size=1)
        self.sub_joy_teleop = rospy.Subscriber("/e_stop_joy_teleop", Bool, self.cb_joy, queue_size=1)

        self.pub_estop = rospy.Publisher('/e_stop', Bool, queue_size=1)

        rospy.loginfo("[%s] Done Initialize " % (self.node_name))


    def cb_xbee(self, msg):
        rospy.loginfo("xbee stopped = %s" % str(msg.data))
        self.xbee_stopped = msg.data

        self.e_stop_on = self.xbee_stopped or self.lora_stopped or self.joy_stopped
        self.pub_estop.publish(self.e_stop_on)

    def cb_lora(self, msg):
        rospy.loginfo("lora stopped = %s" % str(msg.data))
        self.lora_stopped = msg.data

        self.e_stop_on = self.xbee_stopped or self.lora_stopped or self.joy_stopped
        self.pub_estop.publish(self.e_stop_on)

    def cb_joy(self, msg):
        rospy.loginfo("joy stopped = %s" % str(msg.data))
        self.joy_stopped = msg.data

        self.e_stop_on = self.xbee_stopped or self.lora_stopped or self.joy_stopped
        self.pub_estop.publish(self.e_stop_on)


if __name__ == "__main__":
    rospy.init_node("estop_node", anonymous=False)
    e_stop = E_stop()
    rospy.spin()

