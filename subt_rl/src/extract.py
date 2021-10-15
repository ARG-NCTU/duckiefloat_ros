#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import message_filters
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
import numpy as np
import time
import pickle as pkl
import warnings


class Process(object):
    def __init__(self):
        self.l1_library = {'fit': [], 'mm': [], 'vae': [], "cgan": []}

        sub_laser = message_filters.Subscriber('/husky1/RL/scan', LaserScan)
        sub_mm = message_filters.Subscriber(
            '/husky1/RL/scan_mmwave', LaserScan)
        sub_vae = message_filters.Subscriber(
            '/husky1/RL/denoised_mmwave', LaserScan)
        sub_cgan = message_filters.Subscriber(
            '/husky1/RL/denoised_cgan', LaserScan)

        ts = message_filters.ApproximateTimeSynchronizer(
            [sub_laser, sub_mm, sub_vae, sub_cgan], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.cb_laser)

        rospy.loginfo("laser_preprocess init")
    
    def curve_fit(self, mm, order = 8):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            indx = np.where(mm<4.9)[0]
            if len(indx)==0: return mm
            
            try:
                z = np.polyfit(indx, mm[indx], order)
            except:
                return mm
            
            xp = np.linspace(0, 240, 241)
            p = np.poly1d(z)

            fit = np.clip(p(xp), 0, 5)

            return fit
    
    def l1(self, laser, inputL):
        l1 = np.mean(np.abs(laser-inputL))
        print(l1)

        return l1

    def cb_laser(self, laser, mm, vae, cgan):
        laser = np.array(laser.ranges)
        mm = np.array(mm.ranges)
        vae = np.array(vae.ranges)
        cgan = np.array(cgan.ranges)
        fit = self.curve_fit(mm)

        self.l1_library["mm"].append(self.l1(laser, mm))
        self.l1_library["fit"].append(self.l1(laser, fit))
        self.l1_library["vae"].append(self.l1(laser, vae))
        self.l1_library["cgan"].append(self.l1(laser, cgan))

        print('------------------')
        
    def on_shutdown(self):
        for key, value in self.l1_library.items():
            print(key, np.mean(value))


if __name__ == "__main__":
    rospy.init_node('laser_mmwave_process')
    process = Process()
    rospy.on_shutdown(process.on_shutdown)
    rospy.spin()
