# this is the three dimensional configuration space for rrt
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yue qi
"""
import numpy as np
# from utils3D import OBB2AABB

def R_matrix(z_angle,y_angle,x_angle):
    # s angle: row; y angle: pitch; z angle: yaw
    # generate rotation matrix in SO3
    # RzRyRx = R, ZYX intrinsic rotation
    # also (r1,r2,r3) in R3*3 in {W} frame
    # used in obb.O
    # [[R p]
    # [0T 1]] gives transformation from body to world 
    return np.array([[np.cos(z_angle), -np.sin(z_angle), 0.0], [np.sin(z_angle), np.cos(z_angle), 0.0], [0.0, 0.0, 1.0]])@ \
           np.array([[np.cos(y_angle), 0.0, np.sin(y_angle)], [0.0, 1.0, 0.0], [-np.sin(y_angle), 0.0, np.cos(y_angle)]])@ \
           np.array([[1.0, 0.0, 0.0], [0.0, np.cos(x_angle), -np.sin(x_angle)], [0.0, np.sin(x_angle), np.cos(x_angle)]])

class env():
    def __init__(self, xmin=0, ymin=0, zmin=0, xmax=1, ymax=1, zmax=2*np.pi, resolution=1):
    # def __init__(self, xmin=-5, ymin=0, zmin=-5, xmax=10, ymax=5, zmax=10, resolution=1):  
        self.resolution = resolution
        self.boundary = np.array([xmin, ymin, zmin, xmax, ymax, zmax]) 
        # self.blocks = getblocks()
        # self.AABB = getAABB2(self.blocks)
        # self.AABB_pyrr = getAABB(self.blocks)
        # self.balls = getballs()
        # self.OBB = np.array([obb([5.0,7.0,2.5],[0.5,2.0,2.5],R_matrix(135,0,0)),
        #                      obb([12.0,4.0,2.5],[0.5,2.0,2.5],R_matrix(45,0,0))])
        self.start = np.array([.5, .99, .25*np.pi])
        self.goal = np.array([.5, 0.1, np.pi])
        self.t = 0 # time 

    # def New_block(self):
        # newblock = add_block()
        # self.blocks = np.vstack([self.blocks,newblock])
        # self.AABB = getAABB2(self.blocks)
        # self.AABB_pyrr = getAABB(self.blocks)

    def move_start(self, x):
        self.start = x

if __name__ == '__main__':
    newenv = env()
