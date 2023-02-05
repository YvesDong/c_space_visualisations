# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from skimage import measure


def CreateSphere(center, r):
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    x, y, z = r * x + center[0], r * y + center[1], r * z + center[2]
    return (x, y, z)


def draw_Spheres(ax, balls):
    for i in balls:
        (xs, ys, zs) = CreateSphere(i[0:3], i[-1])
        ax.plot_wireframe(xs, ys, zs, alpha=0.15, color="b")


def draw_block_list(ax, blocks, color=None, alpha=0.15):
    '''
    drawing the blocks on the graph
    '''
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                 dtype='float')
    f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    n = blocks.shape[0]
    d = blocks[:, 3:6] - blocks[:, :3]
    vl = np.zeros((8 * n, 3))
    fl = np.zeros((6 * n, 4), dtype='int64')
    for k in range(n):
        vl[k * 8:(k + 1) * 8, :] = v * d[k] + blocks[k, :3]
        fl[k * 6:(k + 1) * 6, :] = f + k * 8
    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        h = ax.add_collection3d(Poly3DCollection(vl[fl], facecolors='black', alpha=alpha, linewidths=1, edgecolors='k'))
        return h


def obb_verts(obb):
    # 0.017004013061523438 for 1000 iters
    ori_body = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], \
                         [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]])
    # P + (ori * E)
    ori_body = np.multiply(ori_body, obb.E)
    # obb.O is orthornormal basis in {W}, aka rotation matrix in SO(3)
    verts = (obb.O @ ori_body.T).T + obb.P
    return verts


# def draw_obb(ax, OBB, color=None, alpha=0.15):
#     f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
#     n = OBB.shape[0]
#     vl = np.zeros((8 * n, 3))
#     fl = np.zeros((6 * n, 4), dtype='int64')
#     for k in range(n):
#         vl[k * 8:(k + 1) * 8, :] = obb_verts(OBB[k])
#         fl[k * 6:(k + 1) * 6, :] = f + k * 8
#     if type(ax) is Poly3DCollection:
#         ax.set_verts(vl[fl])
#     else:
#         h = ax.add_collection3d(Poly3DCollection(vl[fl], facecolors='black', alpha=alpha, linewidths=1, edgecolors='k'))
#         return h


def draw_line(ax, SET, visibility=1, color=None):
    if SET != []:
        for i in SET:
            xs = i[0][0], i[1][0]
            ys = i[0][1], i[1][1]
            zs = i[0][2], i[1][2]
            line = plt3d.art3d.Line3D(xs, ys, zs, alpha=visibility, color=color)
            ax.add_line(line)


def visualization(params):
    if params.ind % 100 == 0 or params.done:
        #----------- list structure
        # V = np.array(list(params.V))
        # E = params.E
        #----------- end
        # edges = params.E
        Path = np.array(params.Path)
        start = params.env.start
        goal = params.env.goal
        # edges = E.get_edge()
        #----------- list structure
        edges = []
        for i in params.Parent:
            edges.append([i,params.Parent[i]])
        #----------- end

        # generate axis objects
        ax = plt.subplot(111, projection='3d')
        
        # ax.view_init(elev=0.+ 0.03*initparams.ind/(2*np.pi), azim=90 + 0.03*initparams.ind/(2*np.pi))
        # ax.view_init(elev=0., azim=90.)
        # ax.view_init(elev=65., azim=60.)
        # ax.view_init(elev=-8., azim=180)
        ax.clear()
        # drawing objects
        # draw_Spheres(ax, initparams.env.balls)
        # draw_block_list(ax, initparams.env.blocks)
        # if initparams.env.OBB is not None:
        #     draw_obb(ax, initparams.env.OBB)
        draw_block_list(ax, np.array([params.env.boundary]), alpha=0)
        draw_line(ax, edges, visibility=0.75, color='g')
        draw_line(ax, Path, color='r')
        # if len(V) > 0:
        #     ax.scatter3D(V[:, 0], V[:, 1], V[:, 2], s=2, color='g', )
        ax.plot(start[0:1], start[1:2], start[2:], 'go', markersize=7, markeredgecolor='k')
        ax.plot(goal[0:1], goal[1:2], goal[2:], 'ro', markersize=7, markeredgecolor='k')
        # adjust the aspect ratio
        ax.dist = 15
        set_axes_equal(ax)
        make_transparent(ax)
        #plt.xlabel('s')
        #plt.ylabel('y')
        ax.set_axis_off()
        plt.pause(0.0001)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def make_transparent(ax):
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

if __name__ == '__main__':
    pass
