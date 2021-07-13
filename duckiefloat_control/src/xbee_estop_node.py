#!/usr/bin/env python

import rospy
import serial
import struct
import math
import time
import copy
from subt_msgs.srv import stop
from std_msgs.msg import Bool

class E_stop(object):
    def __init__(self):
        # 0x 7E 00 0B 88 01 49 53 00 01 00 02 00 00 00 D5
        self.HIGH_STATE = bytearray([126, 0, 11, 136, 1, 73, 83, 0, 1, 0, 2, 0, 0, 2, 213])
        # 0x 7E 00 0B 88 01 49 53 00 01 00 02 00 00 00 D7
        self.LOW_STATE = bytearray([126, 0, 11, 136, 1, 73, 83, 0, 1, 0, 2, 0, 0, 0, 215])
        ESTOP_STATE = False

        self.PORT = rospy.get_param("~port","/dev/ftdi_DN05LFGP")
        rospy.loginfo("port %s"%self.PORT)
        self.serial_link = serial.Serial(self.PORT, 9600)

        self.pub_estop = rospy.Publisher('/e_stop_xbee', Bool, queue_size=1)
        
        self.timer = rospy.Timer(rospy.Duration(1), self.estop_monitor)

        rospy.loginfo("xbee e_stop initiailized")



    def estop_monitor(self, event):
        # note = 0x 7E 00 04 08 01 49 53 5A
        note = bytearray([126, 0, 4, 8, 1, 73, 83, 90])
        #rospy.loginfo( "Write to xbee: %s" % str(list(note)))
        self.serial_link.write(note)
        time.sleep(0.1)

        status = self.serial_link.read(self.serial_link.inWaiting())
        #rospy.loginfo( "Serial status: %s" % str(list(status)))

        if status == self.LOW_STATE:
            rospy.loginfo("Normal State")
            self.pub_estop.publish(False)

        elif status == self.HIGH_STATE:
            rospy.logwarn("Emergency Stop !!!")
            self.pub_estop.publish(True)
        
        else:
            print "Other State"

if __name__ == "__main__":
    rospy.init_node("xbee_estop_node", anonymous=False)
    rospy.loginfo("[%s] Initializing " % (rospy.get_name()))

    e_stop = E_stop()
    rospy.spin()

