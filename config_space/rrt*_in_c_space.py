from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import polygon_tools as poly
import robot_tools
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import copy
import argparse
from plot_tools.surf_rotation_animation import TrisurfRotationAnimator
# import latex
from rrt_3D.env3D import env
from rrt_3D.utils3D import getDist, sampleFree, nearest, steer, isCollide, near, visualization, cost, path
from rrt_3D.rrt_star3D import rrtstar
# from rrt_3D.FMT_star3D import *
from rrt_3D.FMT_star3D_potential import *
from rrt_3D.plot_util3D import *
""" 

Plot an example of config space for Autonomous Mobile Robots lecture notes 

Requires: numpy, matplotlib, argparse, scikit-image (>=0.13, for marching cubes)

Author: Nicholas Lawrance (nicholas.lawrance@mavt.ethz.ch)

"""

plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Basic visualisation of configuration space for mobile robot')
parser.add_argument('-nx', type=int, default=40, help='Resolution (n points in each dimension')
parser.add_argument('-rf', '--robot-footprint', default='config/bar_robot.csv', help='Robot footprint csv file')
parser.add_argument('-no', '--n-obstacles', type=int, default=2, help='Number of obstacles')
parser.add_argument('-ns', '--n-samples', type=int, default=5, help='Number of sample locations for testing')
parser.add_argument('-ss', '--std-samples', type=float, default=0.1, help='Sample standard deviation')
parser.add_argument('--seed', type=int, default=5, help='Numpy random seed')
parser.add_argument('--animation', action='store_true', help='Generate animation')
args = parser.parse_args()

nx = args.nx
num_obstacles = args.n_obstacles
n_obs_samples = args.n_samples
obs_std = args.std_samples
np.random.seed(args.seed)

# Generate obstacles (random points then convex hull)
# obs_centres = [poly.Point(*np.random.uniform(size=2)) for i in range(num_obstacles)]
# print("obs_centres ", obs_centres)
# obstacles = []
# for pc in obs_centres:
#     px, py = np.random.normal(pc, obs_std, size=(n_obs_samples, 2)).T
#     px, py = np.clip(px, 0.0, 1.0), np.clip(py, 0.0, 1.0)
#     p = poly.PointList([poly.Point(x, y) for x, y in zip(px, py)])
#     p = poly.convex_hull(p)
#     obstacles.append(p)

o = np.array([[.5, .3, .25, .5, .7, .75], [.3, .6, .55, .3, .6, .55]])
obstacles = []
for i in range(num_obstacles):
    # px, py = np.random.normal(pc, obs_std, size=(n_obs_samples, 2)).T
    px, py = o[0, 3*i:3*i+3], o[1, 3*i:3*i+3]
    # px, py = np.clip(px, 0.0, 1.0), np.clip(py, 0.0, 1.0)
    p = poly.PointList([poly.Point(x, y) for x, y in zip(px, py)])
    p = poly.convex_hull(p)
    obstacles.append(p)

# Get some random points and see if they're in the obstacles:
in_obs, out_obs = poly.PointList([]), poly.PointList([])
for i in range(200):
    p = poly.Point(*np.random.uniform(size=2))
    collision = False
    for o in obstacles:
        if o.point_inside(p):
            collision = True
            break
    if collision:
        in_obs.append(p)
    else:
        out_obs.append(p)

# plot workspace sampling
f1, a1 = plt.subplots()
h_obs = []
for o in obstacles:
    h_obs.append(PlotPolygon(o, color='lightgrey', zorder=1))
c_obs = PatchCollection(h_obs)
a1.add_collection(c_obs)
a1.scatter(*zip(*in_obs), color='r', marker='x')
a1.scatter(*zip(*out_obs), color='g', marker='.')
print("Intersect: {0}".format(obstacles[0].intersect(obstacles[1])))

# Load the robot shape
robo = robot_tools.Robot2D(footprint_file=args.robot_footprint)

# Now try robot poses:
# a1.add_artist(PlotPolygon(robo.get_current_polygon(), facecolor='r'))

robo.set_position((0.25, 0.38))
robo.get_current_polygon().intersect(obstacles[-1])

x, y, h = np.linspace(0, 1, nx), np.linspace(0, 1, nx), np.linspace(0, 2*np.pi, nx)
v = np.zeros((len(x), len(y), len(h))) # 3D C-space - 1:blocked; 0:free
for i,xi in enumerate(x):
    for j, yj in enumerate(y):
        robo.set_position((xi, yj))
        for k, hk in enumerate(h):
            in_obs = 0.0
            robo.set_heading(hk)
            fp = robo.get_current_polygon()
            for o in obstacles:
                if fp.intersect(o):
                    in_obs = 1.0
                    break
            v[i, j, k] = in_obs

# plot 2*2 c-space construction
robo.set_position([0.2, 0.2])
f2, a2 = plt.subplots(2, 2)
for i, ax in enumerate(a2.flat):
    dex = int(i*0.25*(len(h)-1))
    ax.matshow(v[:, :, dex].transpose(), origin='lower', extent=[0, 1, 0, 1], cmap='Greys')
    ax.add_collection(PatchCollection(copy.copy(h_obs)))
    robo.set_heading(h[dex])
    ax.add_artist(PlotPolygon(robo.get_current_polygon(), facecolor='r'))
    ax.plot(*robo.position, color='g', marker='x')
    ax.set_title(r"$\theta = {0:0.1f}$".format(h[dex]*180/np.pi))
    ax.tick_params(top=0, left=0)

# run rrt
# p = rrtstar(v, nx)
# p.run()
rrt = FMT_star(v, nx, radius = 1, n = 600)
rrt.FMTrun()

# plot RRT optimal path
# TODO: wrong overlapping - https://stackoverflow.com/questions/13932150/matplotlib-wrong-overlapping-when-plotting-two-3d-surfaces-on-the-same-axes
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

edges = []
for i in rrt.Parent:
    edges.append([i, rrt.Parent[i]])
start = rrt.env.start
goal = rrt.env.goal
draw_line(ax, edges, visibility=0.75, color='g')
draw_line(ax, rrt.Path, color='r')
ax.plot3D(start[0], start[1], start[2], 'go', markersize=7, markeredgecolor='k')
ax.plot3D(goal[0], goal[1], goal[2], 'ro', markersize=7, markeredgecolor='k')

# plot obstacle surface
# TODO: donut instead of cube in RRT
verts, faces, normals, values = measure.marching_cubes(v, spacing=(x[1]-x[0], y[1]-y[0], h[1]-h[0]))
ax_lims = [[0, x[-1]], [0, y[-1]], [0, h[-1]]]
ax.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2], edgecolor='k')
# ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
ax.set_xlim(ax_lims[0])
ax.set_ylim(ax_lims[1])
ax.set_zlim(ax_lims[2])
ax.set_xlabel(r'$x_c$')
ax.set_ylabel(r'$y_c$')
ax.set_zlabel(r"$\theta (rad)$")
plt.show()

# plot escape animation
ax1 = plt.subplot(111)
i = len(rrt.Path)
while i >= 0:
    j = 1
    k = i - 1
    if i == 0:
        j = 0
        k = 0
    currAngle = rrt.Path[k][j][2]
    currPos = rrt.Path[k][j][:2]
    dex = int(currAngle/(2*np.pi)*(len(h)-1))
    ax1.clear()
    ax1.matshow(v[:, :, dex].transpose(), origin='lower', extent=[0, 1, 0, 1], cmap='Greys')
    ax1.add_collection(PatchCollection(copy.copy(h_obs)))
    robo.set_position(currPos)
    robo.set_heading(currAngle)
    ax1.add_artist(PlotPolygon(robo.get_current_polygon(), facecolor='r'))
    ax1.plot(*robo.position, color='g', marker='x')
    ax1.set_title(r"$\theta = {0:0.1f}$ rad".format(currAngle))
    ax1.tick_params(top=0, left=0)
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim(0, 1)
    plt.pause(2)
    i -= 1
plt.show()

# escape shaded map
ax2 = plt.subplot(111)
ax2.set_title(r"$\theta = {0:0.1f}$ rad".format(currAngle))
ax2.add_collection(PatchCollection(copy.copy(h_obs)))
ax2.tick_params(top=0, left=0)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
i = len(rrt.Path)
while i >= 0:
    j = 1
    k = i - 1
    if i == 0:
        j = 0
        k = 0
    currAngle = rrt.Path[k][j][2]
    currPos = rrt.Path[k][j][:2]
    robo.set_position(currPos)
    robo.set_heading(currAngle)
    ax2.add_artist(PlotPolygon(robo.get_current_polygon(), facecolor='r', alpha=1-i/(len(rrt.Path)+1)))
    ax2.plot(*robo.position, color='g', marker='x')
    i -= 1
plt.show()

# if args.animation:
#     rotator = TrisurfRotationAnimator(verts, faces, ax_lims=ax_lims, delta_angle=5.0,
#                                       x_label=r'$x_c$', y_label=r'$y_c$', z_label=r"$\theta (^{\circ})$")
#     ani = animation.FuncAnimation(rotator.f, rotator.update, 72, init_func=rotator.init, interval=10, blit=False)
#     # ani.save('fig/config_space_rotation.gif', writer='imagemagick', fps=15)
#     ani.save('fig/config_space_rotation.mp4', writer='ffmpeg', fps=int(15),
#                        extra_args=["-crf", "18", "-profile:v", "main", "-tune", "animation", "-pix_fmt", "yuv420p"])

