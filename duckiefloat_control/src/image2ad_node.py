#! /usr/bin/env python

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import Tkinter as tk
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray, Float32MultiArray


class Detect(object):
    def __init__(self):
        self.temp = Int16MultiArray()
        self.temp.data = [255,77,76,60,0,0]
        self.i = 0
        self.a = 0
        self.d = 0
        self.pose_c = Float32MultiArray()
        self.pose_t = Float32MultiArray()
        self.rgb_sub = rospy.Subscriber('/rgb', Int16MultiArray, self.save, queue_size = 1)
        self.img_sub = rospy.Subscriber('img', Image, self.Find_angle_dis, queue_size = 10)
        self.Pose_c = rospy.Publisher("/state_estimator_node/current_pose", Float32MultiArray, queue_size = 1)
        self.Pose_t = rospy.Publisher("/state_estimator_node/target_pose", Float32MultiArray, queue_size = 1)
    def save(self, data):
        self.temp = data
    def Find_angle_dis(self,data):
        Img = bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")

       #----------detect blue line------------#
        low = np.array([self.temp.data[3],self.temp.data[4],self.temp.data[5]])
        up = np.array([self.temp.data[0], self.temp.data[1], self.temp.data[2]])
        Img = cv2.inRange(Img, low, up)
        #Img = 255 -Img
    
    #----------optimi----------------#
        kernel = np.ones((3,3), np.uint8)
        Img = cv2.erode(Img, kernel)
        Img = cv2.dilate(Img,kernel)

    #----------find_contour-------------#
        _, contours, hierarchy = cv2.findContours(Img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)


    #----------place box----------------------------#
        blackbox = cv2.minAreaRect(contours[0])
        (x_min, y_min), (w_min, h_min), angle = blackbox #x,y:center local w,h:width height
        if abs((w_min)-(h_min)) < 100:
           blackbox = cv2.minAreaRect(contours[1])
           (x_min, y_min), (w_min, h_min), angle = blackbox
        box = cv2.boxPoints(blackbox)
        box = np.int0(box)
        (x1,y1), (x2,y2), (x3,y3), (x4,y4) = box
        cv2.drawContours(Img,[box],0,(125, 0, 0),3)
        d = x1 - x2
        if w_min > h_min:
             if abs(d) < 10:
                 self.a = 0
                 #self.d = 0
                 s = 'S'
             else:
                 self.a = 90 - abs(angle)
                 #self.d = x_min - 340
                 s = 'L'
        if w_min < h_min:
              if abs(d) < 10:
                 self.a = 0
                 #self.d = 0
                 s = 'S'
              else:
                 self.a = (-1)*(abs(angle))
                 #self.d = x_min - 340
                 s = 'R'
        #print('\n'+s+'_'+str(abs(x_min-340))+'_'+str(self.a))
        #self.pose_c = Int16MultiArray()
        self.pose_c.data = [-self.a, -(x_min-340)]
        rospy.loginfo(str(self.pose_c.data))
        #self.Pose_t = Int16MultiArray()
        self.pose_t.data = [90,0]
        self.Pose_c.publish(self.pose_c)
        self.Pose_t.publish(self.pose_t)
        #cv2.imshow('iii',Img)

    #----------show--------#
        #cv2.imshow('Img',Img)
        self.i += 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            self.i += 1






if __name__ == '__main__':
    rospy.init_node("image2ad", anonymous=True)
    detecter = Detect()
    rospy.spin()
