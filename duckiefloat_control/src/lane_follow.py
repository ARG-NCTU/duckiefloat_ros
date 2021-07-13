#!/usr/bin/env python

import numpy as np
import rospy
#import picamera
import math
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import cv2
import time
import torch
import torch.nn as nn
from torchvision import datasets ,transforms
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt  
import torch.optim as optim
import torchvision

weight_path_angle = "/home/ubuntu/subt-duckiefloat/catkin_ws/src/duckiefloat_control/src/line_angle_71.pth"
weight_path_distance = "/home/ubuntu/subt-duckiefloat/catkin_ws/src/duckiefloat_control/src/line_center_83.pth"

class Temp_Model_angle(nn.Module):
    def __init__(self):
        super(Temp_Model_angle, self).__init__()
        self.conv1 = nn.Sequential(              
            nn.Conv2d(
                in_channels=3,              
                out_channels=32,            
                kernel_size=4,              
                stride=1,                   
                padding=0,                  
            ),                                                 
            nn.MaxPool2d(kernel_size=2, stride=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=1,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.fc1 = nn.Linear(38528, 200)
        self.fc2 = nn.Linear(200, 13)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Temp_Model_distance(nn.Module):
    def __init__(self):
        super(Temp_Model_distance, self).__init__()
        self.conv1 = nn.Sequential(              
            nn.Conv2d(
                in_channels=3,              
                out_channels=32,            
                kernel_size=4,              
                stride=1,                   
                padding=0,                  
            ),                                                 
            nn.MaxPool2d(kernel_size=2, stride=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=1,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.fc1 = nn.Linear(38528, 200)
        self.fc2 = nn.Linear(200, 10)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Pose(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.initial()
        self.count = 0
        self.dcc = 0.16
        self.state = None
        self.Angle = np.array([-30,-40,-50,-60,-70,-80,30,40,50,60,70,80,0]) #L-----R, S
        self.Distance = np.array([-70,-140,-210,-280,-350,70,140,210,280,350]) #L-----R
        self.Pose_c = rospy.Publisher("/state_estimator_node/current_pose", Float32MultiArray, queue_size = 1)
        self.Pose_t = rospy.Publisher("/state_estimator_node/target_pose", Float32MultiArray, queue_size = 1)
        self.img_sub = rospy.Subscriber('img', Image, self.c_pose, queue_size = 1)

    def initial(self):
        self.model_angle = Temp_Model_angle()
        self.model_distance = Temp_Model_distance()
        self.model_angle.load_state_dict(torch.load(weight_path_angle))
        self.model_distance.load_state_dict(torch.load(weight_path_distance))

    def c_pose(self, data):
        self.count += 1
        if self.count == 1:
            self.count = 0
            try:
                # convert image_msg to cv format
                img = bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")
                #img = cv2.resize(img, self.dim)

                data_transform = transforms.Compose([
                    transforms.ToTensor()])
                img = data_transform(img)
                images = torch.unsqueeze(img,0)
                
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                images = images.to(device)
                self.model_angle = self.model_angle.to(device)
                self.model_distance = self.model_distance.to(device)

                output_angle = self.model_angle(images)
                output_distance = self.model_distance(images)
                top1 = output_angle.argmax()
                top2 = output_distance.argmax()
                self.angle = self.Angle[top1]
                self.distance = self.Distance[top2]
                
                # pose estimate
                current_pose = Float32MultiArray()
                target_pose = Float32MultiArray()
                # L & left(d<0)
                if (self.angle < 0) and (self.distance < 0):
                    self.distance = (-1)*(self.distance*math.sin(math.radians(self.angle)) - self.dcc*math.cos(math.radians(self.angle)))
                    self.angle = 90 + self.angle
                    self.state = 'L'+str(self.angle)+'D'+str(self.distance)
                # L & right(d>0)
                if (self.angle < 0) and (self.distance > 0):
                    self.distance = (-1)*self.distance*math.sin(math.radians(self.angle)) + self.dcc*math.cos(math.radians(self.angle))
                    self.angle = 90 + self.angle
                    self.state = 'L'+str(self.angle)+'D'+str(self.distance)
                # R & left(d<0)
                if (self.angle > 0) and (self.distance < 0):
                    self.distance = (-1)*((-1)*self.distance*math.sin(math.radians(self.angle)) + self.dcc*math.cos(math.radians(self.angle)))
                    self.angle = 90 - self.angle
                    self.state = 'R'+str(self.angle)+'D'+str(self.distance)
                # R & right(d>0)
                if (self.angle > 0) and (self.distance > 0):
                    self.distance = self.distance*math.sin(math.radians(self.angle)) - self.dcc*math.cos(math.radians(self.angle))
                    self.angle = 90 - self.angle
                    self.state = 'R'+str(self.angle)+'D'+str(self.distance)
                # S
                if self.angle == 0:
                    self.distance = self.distance
                    self.angle = self.angle
                    self.state = 'S'+str(self.angle)+'D'+str(self.distance)

                current_pose[0] = self.distance
                current_pose[1] = self.angle
                target_pose[0] = 0
                tarhet_pose[1] = 0
 
                self.Pose_c.publish(current_pose)
                self.Pose_t.publish(target_pose)
                
                rospy.loginfo('\n'+self.state+'\n')

            except CvBridgeError as e:
                print(e)

if __name__ == "__main__":
    rospy.init_node("lane_follow", anonymous=False)
    POSE = Pose()
    rospy.spin()
