#!/usr/bin/env python
import rospy
import math
from motor_hat_driver import MotorHatDriver
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray

class ControlNode(object):
	def __init__(self):
		self.node_name = rospy.get_name()
		self.motors = MotorHatDriver()
		#get twist msg
		self.sub_cmd_vel = rospy.Subscriber("/cmd_vel", Twist, self.cmdCallback, queue_size = 1)
		#send pwm hat data to "pca9685_activity.cpp"
		self.pub_cmd_brushless = rospy.Publisher("/brushless_command", Int32MultiArray, queue_size = 1)

	def cmdCallback(self, msg):
		brushless_msg = Int32MultiArray()
		raw_linear_x = msg.linear.x #DC
		raw_linear_z = msg.linear.z	#brushless
		raw_angular_z = msg.angular.z #DC
		#angulat_z > 0 == left / angular_z < 0 == right

		if(raw_linear_x >= 0):

		#flank brushless motors speed control
			motor_flank = 4100+2600*raw_linear_z #+4100 to set brushless motor always on
			brushless_msg.data = [motor_flank, motor_flank, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
			self.pub_cmd_brushless.publish(brushless_msg)

		#linear.x and angular.z DC motors control
			#m1~m4 correspond to the m1~m4 mark on the motor hat
			m1 = 0	#negative angular.z / turn right
			m2 = 0	#linear.x
			m3 = 0	#positive angular.z / turn left
			m4 = 0	#linear.x
			
			#linear.x DC motors
			m2 = raw_linear_x*0.7
                        m4 = raw_linear_x*0.7 #to limit the max voltage on motors, for 3S battetry

			#angular.z DC motors
			motor_rear = raw_angular_z*0.75 #to limit the max voltage on motors, for 3S battetry
			#turn left
			if(motor_rear > 0):
				m3 = -motor_rear
			
			#turn right
			elif(motor_rear < 0):
				m1 = motor_rear

			elif(motor_rear == 0):
				m1 = 0
				m3 = 0

			self.motors.setMotorSpeed(m1, m2, m3, m4)

			#rospy.loginfo("Twist linear.z : %f\n", motor_flank)
			#rospy.loginfo("Twist angular.z : %f\n", motor_rear)	
			
	def on_shutdown(self):
        	rospy.logwarn("Stopping brushless motors")
		
		brushless_msg = Int32MultiArray()
		brushless_msg.data = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
		self.pub_cmd_brushless.publish(brushless_msg)

	
if __name__ == "__main__":
	rospy.init_node("control_node", anonymous = True)
	node = ControlNode()
	
	rospy.on_shutdown(node.on_shutdown)
	rospy.spin()


