#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from simple_pid import PID

class AutonomousControlNode(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))

        self.sample_time = 0.1

        self.sub_current_pose = rospy.Subscriber("/state_estimator_node/current_pose", Float32MultiArray, self.cbcp, queue_size = 10)
        self.sub_target_pose = rospy.Subscriber("/state_estimator_node/target_pose", Float32MultiArray, self.cbtp, queue_size = 10)
        self.pub_vel_cmd = rospy.Publisher("autonomous/cmd_vel", Twist, queue_size=1)
        
        self.target_pose = np.zeros(2)
        self.current_pose = np.zeros(2)
        self.memory = [0, 0, 0, 0, 0]

        self.pid = PID(1.2, 0, 3, sample_time = self.sample_time, setpoint = 0, output_limits = [-1, 1])
        self.timer = rospy.Timer(rospy.Duration(self.sample_time), self.control)
        rospy.on_shutdown(self.shutdown)


        rospy.loginfo("done Initializing")
        

    # pose[0]: distance
    # pose[1]: angle
    def cbcp(self, msg):
        # rospy.loginfo("cp cb")
        self.current_pose[0] = msg.data[0]
        self.current_pose[1] = msg.data[1]
    def cbtp(self, msg):
        self.target_pose[0] = msg.data[0]
        self.target_pose[1] = msg.data[1]

    def control(self, event):
        self.pid.set_point = self.target_pose[0] + self.current_pose[1]*1.0
        control = self.pid(self.current_pose[0])
        self.memory.append(control)
        self.memory.pop(0)
        #print self.memory
        # rospy.loginfo(str(len(self.memory)))
        #rospy.loginfo(np.average(self.memory))

        msg = Twist()
        msg.linear.x = 0.5
        msg.angular.z = -np.average(self.memory)
        self.pub_vel_cmd.publish(msg)



    def shutdown(self):
        self.timer.shutdown()
        msg = Twist()
        msg.linear.x = 0
        msg.angular.z = 0
        self.pub_vel_cmd.publish(msg)


if __name__ == '__main__':
    rospy.init_node('autonomous_control_node', anonymous=False)
    node = AutonomousControlNode()
    rospy.spin()
    '''
    vel_count = 0
    start_time = -1
    while not rospy.is_shutdown():
        #get loop time
        if node.time == -1:
            node.time = time.time()
        else:
            node.pid.sample_time = time.time()-node.time
            node.time = time.time()
                print "cur: ", node.current_pose[0], ", ", node.current_pose[1] 
        node.pid.setpoint = node.target_pose[0]#+node.current_pose[1]*0.6
        control = node.pid(node.current_pose[0])
        msg = Twist()
        if abs(node.current_pose[0]-node.pid.setpoint)<0.05:
            msg.linear.x = 0.15
            #if vel_count < 30:
            #   msg.v = 0.2
            #   vel_count+=1
            #else:
            #   msg.v = 0
        elif abs(node.current_pose[0]-node.pid.setpoint)>0.3 and abs(node.current_pose[0]-node.pid.setpoint)<0.4:
            msg.linear.x = -0.4
        elif abs(node.current_pose[0]-node.pid.setpoint)>0.4:
            msg.linear.x = -0.8
        else:
            msg.linear.x = 0.1
            #if vel_count < 30:
            #   msg.v = 0.1
            #   vel_count+=1
            #else:
            #   msg.v = 0
        if vel_count > 30 and rospy.time.now()-start_time > 10:
            start_time = rospy.time.now()
            vel_count = 0
        msg.angular.z = -control
        node.pub_vel_cmd.publish(msg)
    '''
