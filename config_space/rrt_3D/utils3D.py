import numpy as np
from numpy.matlib import repmat
import pyrr as pyrr
from collections import deque

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/")
from rrt_3D.plot_util3D import visualization


def getRay(x, y):
    direc = [y[0] - x[0], y[1] - x[1], y[2] - x[2]]
    return np.array([x, direc])

def getDist(pos1, pos2):
    return np.sqrt(sum([(pos1[0] - pos2[0]) ** 2, (pos1[1] - pos2[1]) ** 2, (pos1[2] - pos2[2]) ** 2]))


''' The following utils can be used for rrt or rrt*,
    required param initparams should have
    env,      environement generated from env3D
    V,        node set
    E,        edge set
    i,        nodes added
    maxiter,  maximum iteration allowed
    stepsize, leaf growth restriction

'''

def sampleFree(params):
    '''biased sampling'''
    p = np.random.uniform(params.env.boundary[0:3], params.env.boundary[3:6])
    # i = np.random.random()
    if isinside(params, p):
        return sampleFree(params)
    else:
        # if i < bias:
        #     return np.array(initparams.xt) + 1
        # else:
        #     return x
        return p

# ---------------------- Collision checking algorithms
def isinside(params, p, thres=.005):
    '''see if inside obstacle'''
    oMap = params.oMap
    normp = (p-params.envLowBound) / (params.envUpBound-params.envLowBound) # normalized p
    oMapShape = oMap.shape

    # limits for iterations
    # TODO: cube neighbor to sphere
    limLow = np.maximum(normp-thres, np.zeros((3)))
    limUp = np.minimum(normp+thres, np.ones((3)))
    limLow = np.floor(limLow*oMapShape).astype(int)
    limUp = np.floor(limUp*oMapShape).astype(int)
    # print(limLow, limUp)
    if np.prod(limUp-limLow) == 0: # no element in the neighbor
        return False
    
    # detect in a neighbor cube
    oMapNeighbor = oMap[limLow[0]:limUp[0],limLow[1]:limUp[1],limLow[2]:limUp[2]]
    if np.amax(oMapNeighbor):
        return True

    return False

def isCollide(params, p1, p2, dist=None, thres=.005):
    '''see if line intersects obstacle'''
    '''specified for expansion in A* 3D lookup table'''
    if dist==None:
        dist = getDist(p1, p2)
    # print('dist: ', dist)
    
    # check in bound
    # TODO: avoid repeat calculation
    nsample = np.ceil(dist/params.normLenth).astype(int)
    # print('nsample: ', nsample)
    vec = np.asarray(p2) - np.asarray(p1)
    for i in range(nsample+1):
        currp = p1 + i/nsample*vec
        if isinside(params, currp, thres=thres):
            return True, dist
    return False, dist

# ---------------------- leaf node extending algorithms
def nearest(initparams, x, isset=False):
    V = np.array(initparams.V)
    if initparams.i == 0:
        return initparams.V[0]
    xr = repmat(x, len(V), 1)
    dists = np.linalg.norm(xr - V, axis=1)
    return tuple(initparams.V[np.argmin(dists)])

def near(initparams, x):
    # s = np.array(s)
    V = np.array(initparams.V)
    if initparams.i == 0:
        return [initparams.V[0]]
    cardV = len(initparams.V)
    eta = initparams.eta
    gamma = initparams.gamma
    # min{γRRT∗ (log(card (V ))/ card (V ))1/d, η}
    r = min(gamma * ((np.log(cardV) / cardV) ** (1/3)), eta)
    if initparams.done: 
        r = 1
    xr = repmat(x, len(V), 1)
    inside = np.linalg.norm(xr - V, axis=1) < r
    nearpoints = V[inside]
    return np.array(nearpoints)

def steer(initparams, x, y, DIST=False):
    # steer from s to y
    if np.equal(x, y).all():
        return x, 0.0
    dist, step = getDist(y, x), initparams.stepsize
    step = min(dist, step)
    increment = ((y[0] - x[0]) / dist * step, (y[1] - x[1]) / dist * step, (y[2] - x[2]) / dist * step)
    xnew = (x[0] + increment[0], x[1] + increment[1], x[2] + increment[2])
    # direc = (y - s) / np.linalg.norm(y - s)
    # xnew = s + initparams.stepsize * direc
    if DIST:
        return xnew, dist
    return xnew, dist

def cost(initparams, x):
    '''here use the additive recursive cost function'''
    if x == initparams.x0:
        return 0
    return cost(initparams, initparams.Parent[x]) + getDist(x, initparams.Parent[x])

def cost_from_set(initparams, x):
    '''here use a incremental cost set function'''
    if x == initparams.x0:
        return 0
    return initparams.COST[initparams.Parent[x]] + getDist(x, initparams.Parent[x])

def path(initparams, Path=[], dist=0):
    x = initparams.xt
    while x != initparams.x0:
        x2 = initparams.Parent[x]
        Path.append(np.array([x, x2]))
        dist += getDist(x, x2)
        x = x2
    return Path, dist

class edgeset(object):
    def __init__(self):
        self.E = {}

    def add_edge(self, edge):
        x, y = edge[0], edge[1]
        if x in self.E:
            self.E[x].add(y)
        else:
            self.E[x] = set()
            self.E[x].add(y)

    def remove_edge(self, edge):
        x, y = edge[0], edge[1]
        self.E[x].remove(y)

    def get_edge(self, nodes = None):
        edges = []
        if nodes is None:
            for v in self.E:
                for n in self.E[v]:
                    # if (n,v) not in edges:
                    edges.append((v, n))
        else: 
            for v in nodes:
                for n in self.E[tuple(v)]:
                    edges.append((v, n))
        return edges

    def isEndNode(self, node):
        return node not in self.E

#------------------------ use a linked list to express the tree 
class Node:
    def __init__(self, data):
        self.pos = data
        self.Parent = None
        self.child = set()

def tree_add_edge(node_in_tree, x):
    # add an edge at the specified parent
    node_to_add = Node(x)
    # node_in_tree = tree_bfs(head, xparent)
    node_in_tree.child.add(node_to_add)
    node_to_add.Parent = node_in_tree
    return node_to_add

def tree_bfs(head, x):
    # searches s in order of bfs
    node = head
    Q = []
    Q.append(node)
    while Q:
        curr = Q.pop()
        if curr.pos == x:
            return curr
        for child_node in curr.child:
            Q.append(child_node)

def tree_nearest(head, x):
    # find the node nearest to s
    D = np.inf
    min_node = None

    Q = []
    Q.append(head)
    while Q:
        curr = Q.pop()
        dist = getDist(curr.pos, x)
        # record the current best
        if dist < D:
            D, min_node = dist, curr
        # bfs
        for child_node in curr.child:
            Q.append(child_node)
    return min_node

def tree_steer(initparams, node, x):
    # steer from node to s
    dist, step = getDist(node.pos, x), initparams.stepsize
    increment = ((node.pos[0] - x[0]) / dist * step, (node.pos[1] - x[1]) / dist * step, (node.pos[2] - x[2]) / dist * step)
    xnew = (x[0] + increment[0], x[1] + increment[1], x[2] + increment[2])
    return xnew

def tree_print(head):
    Q = []
    Q.append(head)
    verts = []
    edge = []
    while Q:
        curr = Q.pop()
       # print(curr.pos)
        verts.append(curr.pos)
        if curr.Parent == None:
            pass
        else:
            edge.append([curr.pos, curr.Parent.pos])
        for child in curr.child:
            Q.append(child)
    return verts, edge

def tree_path(initparams, end_node):
    path = []
    curr = end_node
    while curr.pos != initparams.x0:
        path.append([curr.pos, curr.Parent.pos])
        curr = curr.Parent
    return path


#---------------KD tree, used for nearest neighbor search
class kdTree:
    def __init__(self):
        pass

    def R1_dist(self, q, p):
        return abs(q-p)

    def S1_dist(self, q, p):
        return min(abs(q-p), 1- abs(q-p))

    def P3_dist(self, q, p):
        # cubes with antipodal points
        q1, q2, q3 = q
        p1, p2, p3 = p
        d1 = np.sqrt((q1-p1)**2 + (q2-p2)**2 + (q3-p3)**2)
        d2 = np.sqrt((1-abs(q1-p1))**2 + (1-abs(q2-p2))**2 + (1-abs(q3-p3))**2)
        d3 = np.sqrt((-q1-p1)**2 + (-q2-p2)**2 + (q3+1-p3)**2)
        d4 = np.sqrt((-q1-p1)**2 + (-q2-p2)**2 + (q3-1-p3)**2)
        d5 = np.sqrt((-q1-p1)**2 + (q2+1-p2)**2 + (-q3-p3)**2)
        d6 = np.sqrt((-q1-p1)**2 + (q2-1-p2)**2 + (-q3-p3)**2)
        d7 = np.sqrt((q1+1-p1)**2 + (-q2-p2)**2 + (-q3-p3)**2)
        d8 = np.sqrt((q1-1-p1)**2 + (-q2-p2)**2 + (-q3-p3)**2)
        return min(d1,d2,d3,d4,d5,d6,d7,d8)



if __name__ == '__main__':
    from env3D import env
    import time
    import matplotlib.pyplot as plt
    class rrt_demo:
        def __init__(self):
            self.env = env()
            self.x0, self.xt = tuple(self.env.start), tuple(self.env.goal)
            self.stepsize = 0.5
            self.maxiter = 10000
            self.ind, self.i = 0, 0
            self.done = False
            self.Path = []
            self.V = []

            self.head = Node(self.x0)
        
        def run(self):
            while self.ind < self.maxiter:
                xrand = sampleFree(self) # O(1)
                nearest_node = tree_nearest(self.head, xrand) # O(N)
                xnew = tree_steer(self, nearest_node, xrand) # O(1)
                collide, _ = isCollide(self, nearest_node.pos, xnew) # O(num obs)
                if not collide:
                    new_node = tree_add_edge(nearest_node, xnew) # O(1)
                    # if the path is found
                    if getDist(xnew, self.xt) <= self.stepsize:
                        end_node = tree_add_edge(new_node, self.xt)
                        self.Path = tree_path(self, end_node)
                        break
                    self.i += 1
                self.ind += 1
            
            self.done = True
            self.V, self.E = tree_print(self.head)
            print(self.E)
            visualization(self)
            plt.show()
            

    
    A = rrt_demo()
    st = time.time()
    A.run()
    print(time.time() - st)

        
        


