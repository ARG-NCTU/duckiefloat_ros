#!/usr/bin/env python

import numpy as np
import rospy
#import picamera
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
import cv2
import time
from picamera.array import PiRGBArray

class Cam(object):
    def __init__(self):
        self.i = 0
    rospy.init_node("camera_node",anonymous=True)
    img_pub = rospy.Publisher('img',Image,queue_size=1)
    #camera = picamera.PiCamera()
    cap = cv2.VideoCapture(0)

    while True:
        #ra = PiRGBArray(camera)
        #time.sleep(0.1)
        #camera.capture(ra, format="bgr")
        #img = ra.array
	ret, img = cap.read()

        data = bridge.cv2_to_imgmsg(img,encoding='passthrough')
        img_pub.publish(data)
        rospy.loginfo('publish an image')

        
        #cv2.imshow('img',img)
        k = cv2.waitKey(5)
        if k == 27:
            break

    cv2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = Cam()
    rospy.spin
