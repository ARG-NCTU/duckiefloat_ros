#!/usr/bin/env python

import time
import picamera
import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()
import io


class cam(object):
    def __init__(self):
        self.i = 0
        #self.stream = io.BytesIO()
    stream = io.BytesIO()
    rospy.init_node("camera_node",anonymous=True)
    camera = picamera.PiCamera()
    while True:
         camera.resolution = (100,70)
         #camera.start_preview()
         #img = np.empty((480*640*3),dtype=np.uint8)
         camera.capture(stream,format='jpeg')
         data = np.fromstring(stream.getvalue(),dtype=np.uint8)
         image = cv2.imdecode(data,1)
         Data = bridge.cv2_to_imgmsg(image)
         rospy.loginfo('1')
         cv2.imshow('img',image)
         img_pub = rospy.Publisher('img',Image,queue_size=1)
         #rospy.loginfo('2')
         img_pub.publish(Data)
         #rospy.loginfo('3')
         k = cv2.waitKey(5)
         if k == 27:
            break
    cv2.destroyAllWindows()




if __name__ == "__main__":
    cam = Cam()
    rospy.spin
