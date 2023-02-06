# this is the three dimensional configuration space for rrt
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yue qi
"""
import numpy as np

# def R_matrix(z_angle,y_angle,x_angle):
#     # s angle: row; y angle: pitch; z angle: yaw
#     # generate rotation matrix in SO3
#     # RzRyRx = R, ZYX intrinsic rotation
#     # also (r1,r2,r3) in R3*3 in {W} frame
#     # used in obb.O
#     # [[R p]
#     # [0T 1]] gives transformation from body to world 
#     return np.array([[np.cos(z_angle), -np.sin(z_angle), 0.0], [np.sin(z_angle), np.cos(z_angle), 0.0], [0.0, 0.0, 1.0]])@ \
#            np.array([[np.cos(y_angle), 0.0, np.sin(y_angle)], [0.0, 1.0, 0.0], [-np.sin(y_angle), 0.0, np.cos(y_angle)]])@ \
#            np.array([[1.0, 0.0, 0.0], [0.0, np.cos(x_angle), -np.sin(x_angle)], [0.0, np.sin(x_angle), np.cos(x_angle)]])

class env():
    def __init__(self, xmin=0, ymin=0, zmin=0, xmax=1, ymax=1, zmax=2*np.pi, resolution=1):
        self.resolution = resolution
        self.boundary = np.array([xmin, ymin, zmin, xmax, ymax, zmax]) 
        self.start = np.array([.5, .6, .5*np.pi])
        self.goal = np.array([.5, 0.1, np.pi])
        self.t = 0 # time 

    def move_start(self, x):
        self.start = x

if __name__ == '__main__':
    newenv = env()
