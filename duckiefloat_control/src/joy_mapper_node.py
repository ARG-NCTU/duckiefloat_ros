#!/usr/bin/env python
import rospy
import math 
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32

class JoyMapper(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))


        # status
        self.auto_en = False
        self.height_control_en = False
        self.height_cmd = 0.0
        self.auto_cmd = None
        self.joy_cmd = None
        self.remote_override = False
        self.angular_trim = 0       #angular Z trim, to eliminate rotation cause by brushless
        self.linear_x_trim = 0      #set a fixed forward speed

        # Publishers
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pub_is_auto = rospy.Publisher("is_auto", Bool, queue_size=1)
        self.pub_e_stop = rospy.Publisher("/e_stop_joy", Bool, queue_size=1)
    
        # Subscribers
        self.sub_joy = rospy.Subscriber("joy", Joy, self.cbJoy, queue_size=1)
        self.sub_remote_joy = rospy.Subscriber("remote_joy", Joy, self.cb_remote_joy, queue_size=1)
        self.sub_auto_control = rospy.Subscriber("autonomous/cmd_vel", Twist, self.cb_auto, queue_size=1)
        self.sub_height_control = rospy.Subscriber("altitude_control_node/control", Float32, self.cb_height_control, queue_size=1)

        # Timer
        self.fusion_timer = rospy.Timer(rospy.Duration(0.1), self.command_fusion)
        
        rospy.loginfo("[%s] Initialize done " %(self.node_name))

    def cbJoy(self, msg):
        if not self.remote_override:
            self.processButtons(msg)
            self.processAxes(msg)

    def cb_remote_joy(sel, msg):
        # if remote user overrides
        self.remote_override = msg.buttons[4]
        if self.remote_override:
            self.processButtons(msg)
            self.processAxes(msg)


    def command_fusion(self, event):
        cmd_msg = Twist()
        if self.auto_en:
            if self.auto_cmd == None: return
            cmd_msg = self.auto_cmd
        else:
            if self.joy_cmd == None: return
            cmd_msg = self.joy_cmd
        if self.height_control_en:
            cmd_msg.linear.z = self.height_cmd
        rospy.loginfo("%f %f"%(cmd_msg.angular.z,cmd_msg.linear.x))

        self.pub_cmd_vel.publish(cmd_msg)

    def cb_auto(self, msg):
        cmd_vel = Twist()
        if self.auto_en:
            cmd_vel = msg
            if self.height_control_en:
                 cmd_vel.linear.z = self.height_cmd
            self.pub_cmd_vel.publish(cmd_vel)

    def cb_height_control(self, msg):
        self.height_cmd = max(msg.data, 0)

    # Axis List index of joy.axes array:
    # 0: Left Horizontal (Left +)
    # 1: Left Vertical (Up +)
    # 2: LT (0 ~ 1)
    # 3: Right Horizontal (Left +)
    # 4: Right Vertical (Up +)
    # 5: RT (0 ~ 1)
    # 6: CrossKey Horizontal (Left 1)
    # 7: CrossKey Vertical (Up 1)
    def processAxes(self, joy_msg):
        cmd_msg = Twist()
        cmd_msg.linear.x = joy_msg.axes[4]
        cmd_msg.angular.z = joy_msg.axes[3]
        cmd_msg.linear.z = joy_msg.axes[1]
        self.joy_cmd = cmd_msg
        rospy.loginfo("%s"%(id(cmd_msg)))
        # angular trim
        if joy_msg.axes[6] == -1:
            self.angular_trim -= 0.05
        elif joy_msg.axes[6] == 1:
            self.angular_trim += 0.05

        # linear.x trim
        if joy_msg.axes[7] == -1:
            self.linear_x_trim -= 0.1
        elif joy_msg.axes[7] == 1:
            self.linear_x_trim += 0.1
        
        cmd_msg.angular.z += self.angular_trim
        cmd_msg.linear.x += self.linear_x_trim

    # Button List index of joy.buttons array:
    # 0: A 
    # 1: B 
    # 2: X
    # 3: Y 
    # 4: Left Back 
    # 5: Right Back
    # 6: Back
    # 7: Start
    # 8: Logitek 
    # 9: Left joystick
    # 10: Right joystick
    def processButtons(self, joy_msg):
        # Button A
        if joy_msg.buttons[0]:
            self.height_control_en = not self.height_control_en
            if self.height_control_en:
                rospy.loginfo("enable height control")
            else: 
                rospy.loginfo("disable height control")
        # Button start
        if joy_msg.buttons[7]:
            self.auto_en = True
            rospy.loginfo("start autonomous")
        # Button back
        if joy_msg.buttons[6]:
            self.auto_en = False
            rospy.loginfo("joystick mode")

        ## Left back button
        if (joy_msg.buttons[4] == 1):
            # EStop ON
            self.pub_e_stop.publish(True)
        ## Right back button
        elif (joy_msg.buttons[5] == 1):
            # EStop OFF
            self.pub_e_stop.publish(False)

if __name__ == "__main__":
	rospy.init_node("joy_mapper_node",anonymous=False)
	joy_mapper = JoyMapper()
	rospy.spin()
