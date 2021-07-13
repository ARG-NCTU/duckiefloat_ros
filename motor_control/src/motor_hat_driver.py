#!/usr/bin/env python

from Adafruit_MotorHAT import Adafruit_MotorHAT
from math import fabs, floor

class MotorHatDriver(object):
    MOTOR_FORWARD = Adafruit_MotorHAT.FORWARD
    MOTOR_BACKWARD = Adafruit_MotorHAT.BACKWARD
    MOTOR_RELEASE = Adafruit_MotorHAT.RELEASE

    DEADZONE = 5.e-2

    def __init__(self, addr=0x60):
        self.motor_hat = Adafruit_MotorHAT(addr=addr)
        self.m1 = self.motor_hat.getMotor(1)
        self.m2 = self.motor_hat.getMotor(2)
        self.m3 = self.motor_hat.getMotor(3)
        self.m4 = self.motor_hat.getMotor(4)

        self.m1_speed = 0
        self.m2_speed = 0
        self.m3_speed = 0
        self.m4_speed = 0

    # all values can be given between -1 and 1
    def setMotorSpeed(self, m1=None, m2=None, m3=None, m4=None):        
        if m1 is not None:
            self.m1_speed = min(1, max(-1, m1))
        if m2 is not None:
            self.m2_speed = min(1, max(-1, m2))
        if m3 is not None:
            self.m3_speed = min(1, max(-1, m3))
        if m4 is not None:
            self.m4_speed = min(1, max(-1, m4))
        self.updatePWM()

    def updatePWM(self):
        pwm_m1 = int(floor(fabs(self.m1_speed) * 255))
        pwm_m2 = int(floor(fabs(self.m2_speed) * 255))
        pwm_m3 = int(floor(fabs(self.m3_speed) * 255))
        pwm_m4 = int(floor(fabs(self.m4_speed) * 255))
        if fabs(self.m1_speed) < self.DEADZONE:
            m1_mode = self.MOTOR_RELEASE
        elif self.m1_speed > 0:
            m1_mode = self.MOTOR_FORWARD
        elif self.m1_speed < 0:
            m1_mode = self.MOTOR_BACKWARD

        if fabs(self.m2_speed) < self.DEADZONE:
            m2_mode = self.MOTOR_RELEASE
        elif self.m2_speed > 0:
            m2_mode = self.MOTOR_FORWARD
        elif self.m2_speed < 0:
            m2_mode = self.MOTOR_BACKWARD

        if fabs(self.m3_speed) < self.DEADZONE:
            m3_mode = self.MOTOR_RELEASE
        elif self.m3_speed > 0:
            m3_mode = self.MOTOR_FORWARD
        elif self.m3_speed < 0:
            m3_mode = self.MOTOR_BACKWARD

        if fabs(self.m4_speed) < self.DEADZONE:
            m4_mode = self.MOTOR_RELEASE
        elif self.m4_speed > 0:
            m4_mode = self.MOTOR_FORWARD
        elif self.m4_speed < 0:
            m4_mode = self.MOTOR_BACKWARD

        self.m1.setSpeed(pwm_m1)
        self.m1.run(m1_mode)
        self.m2.setSpeed(pwm_m2)
        self.m2.run(m2_mode)
        self.m3.setSpeed(pwm_m3)
        self.m3.run(m3_mode)
        self.m4.setSpeed(pwm_m4)
        self.m4.run(m4_mode)

    def __del__(self):
        self.m1.run(self.MOTOR_RELEASE)
        self.m2.run(self.MOTOR_RELEASE)
        self.m3.run(self.MOTOR_RELEASE)
        self.m4.run(self.MOTOR_RELEASE)
        del self.motor_hat
