#!/usr/bin/env python
import rospy
import math
from motor_hat_driver import MotorHatDriver
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float64

class ControlNode(object):
	def __init__(self):
		self.node_name = rospy.get_name()
		self.motors = MotorHatDriver()
		self.sub_cmd_vel = rospy.Subscriber("/cmd_vel", Twist, self.cmdCallback, queue_size = 1)
        
		self.pub_cmd_brushless = rospy.Publisher("/brushless_command", Int32MultiArray, queue_size = 1)
		self.pub_cmd_dynamixel = rospy.Publisher("/tilt_controller/command", Float64, queue_size = 1) 

	def cmdCallback(self, msg):
		brushless_msg = Int32MultiArray()
		dynamixel_msg = Float64()
		raw_linear_x = msg.linear.x*0.5 #limit the forward speed
		raw_linear_z = msg.linear.z
		raw_angular_z = msg.angular.z
		#angulat_z > 0 == left / angular_z < 0 == right
		

		if(raw_linear_x >= 0):
	#dynamixel servo angle
			raw_angle_pitch = math.atan2(raw_linear_z, raw_linear_x)
			#mapping_angular = raw_angle_pitch/3.1415926*180*31 + 5200
			#mapping_angular_rev = 5200 - raw_angle_pitch/3.1415926*180*31
			mapping_angular = -(math.pi*1/2-raw_angle_pitch)
			dynamixel_msg.data = max( min( mapping_angular, math.pi*1/3), -math.pi*1/3)
                        if(raw_linear_z < 0.02):
                            dynamixel_msg.data = 0
                        self.pub_cmd_dynamixel.publish(dynamixel_msg)

	#flank brushless motor control
			motor_flank = 4100+2600*math.fabs(math.sqrt((math.pow(raw_linear_z, 2) + math.pow(raw_linear_x, 2))))
			brushless_msg.data = [motor_flank, motor_flank, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
			self.pub_cmd_brushless.publish(brushless_msg)
	#rear DC motor control
			left_x = 0
			left_up = 0
			right_x = 0
			right_up = 0
			motor_rear = raw_angular_z

			if(motor_rear > 0):
				#left
				motor_rear = math.fabs(motor_rear)*-1*0.75
				right_x = motor_rear
				self.motors.setMotorSpeed(left_x, left_up, right_x, right_up)

			elif(motor_rear < 0):
				#right
				motor_rear = math.fabs(motor_rear)*0.75
				left_x = motor_rear
				self.motors.setMotorSpeed(left_x, left_up, right_x, right_up)

			elif(motor_rear == 0 or raw_angle_pitch == 0):
				self.motors.setMotorSpeed(left_x, left_up, right_x, right_up)

			#self.motors.setMotorSpeed(left, flank, right, flank)

		#	rospy.loginfo("flank speed : %f\n", motor_flank)
		#	rospy.loginfo("rear speed: %f\n", motor_rear)
		#	rospy.loginfo("pitch angle: %f\n", mapping_angular)
                
                #it seems that this method doesn't work, but I keep it
                if rospy.is_shutdown():
                    brushless_msg.data = [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                    self.pub_cmd_brushless.publish(brushless_msg)
			


	
if __name__ == "__main__":
	rospy.init_node("control_node", anonymous = True)
	node = ControlNode()
	rospy.spin()


