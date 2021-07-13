#! /usr/bin/env python
import rospy
import Tkinter as tk
from std_msgs.msg import String
from std_msgs.msg import UInt16, Int16MultiArray

pub = rospy.Publisher('/rgb', Int16MultiArray, queue_size=10)

def set_B(v):
    global k
    k = int(v)

def set_G(v):
    global j
    j = int(v)

def set_R(v):
    global i
    i = int(v)

def set_b(v):
    global K
    K = int(v)

def set_g(v):
    global J
    J = int(v)

def set_r(v):
    global I
    I = int(v)

def PUB():
    global k,j,i,K,J,I
    a = Int16MultiArray()
    a.data = [k,j,i,K,J,I]
    pub.publish(a)

def talker():
  rospy.init_node('pump_controller', anonymous=True)
  rate = rospy.Rate(10)



root = tk.Tk()

s1 = tk.Scale(root,label='B_up', from_=0, to=255, orient="horizontal",length=600, showvalue=1,tickinterval=15, resolution=1, command=set_B)
s1.pack()

s2 = tk.Scale(root,label='G_up', from_=0, to=255, orient="horizontal",length=600, showvalue=1,tickinterval=15, resolution=1, command=set_G)
s2.pack()

s3 = tk.Scale(root,label='R_up', from_=0, to=255, orient="horizontal",length=600, showvalue=1,tickinterval=15, resolution=1, command=set_R)
s3.pack()

s4 = tk.Scale(root,label='B_low', from_=0, to=255, orient="horizontal",length=600, showvalue=1,tickinterval=15, resolution=1, command=set_b)
s4.pack()

s5 = tk.Scale(root,label='G_low', from_=0, to=255, orient="horizontal",length=600, showvalue=1,tickinterval=15, resolution=1, command=set_g)
s5.pack()

s6 = tk.Scale(root,label='R_low', from_=0, to=255, orient="horizontal",length=600, showvalue=1,tickinterval=15, resolution=1, command=set_r)
s6.pack()

btnPumpStart = tk.Button(root, text="RGB PUB", command=PUB)
btnPumpStart.pack()

if __name__ == '__main__':
    k = 0
    j = 0
    i = 0
    K = 0
    I = 0
    J = 0
    try:
        talker()
        root.mainloop()
    except rospy.ROSInterruptException:
        pass
