'''
VOXELMAP 
Mapping arrays to voxel models
Andrew Garcia, 2022

'''
import numpy as np
import random as ran
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib import colors

import pickle


def load_array(filename):
    return pickle.load( open(filename, "rb" ),encoding='latin1')

def save_array(tensor,filename):
    return pickle.dump(tensor,open(filename,'wb'))


def binarize(tensor,colorindex=1):
    '''convert a tensor with integer entries to either 0 if 0 or 1 if !=0'''
    tensorv = np.zeros((tensor.shape))
    for i in np.argwhere(tensor!=0):
        tensorv[tuple(i)] = colorindex
    
    return tensorv

    
def set_axes_radius(ax, origin, radius):
    '''set_axes_radius and set_axes_equal * * * Credit:
    Mateen Ulhaq (answered Jun 3 '18 at 7:55)
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to'''
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)



class Model:

    def __init__(self,array,colormap=cm.cool,cmap_alpha=1):
        self.array = array          # array of third-order (3-D)
        self.hashblocks = {}        #hashblocks is a dictionary for the block colormap (see customadd)
        self.colormap = colormap
        self.cmap_alpha = cmap_alpha

    def customadd(self, key, color, alpha=1):
        '''Make your own 3-D colormap option. Adds to hashblocks dictionary.

        Parameters
        ----------
        key : int   
            array value to color as voxel
        color : str
            color of voxel with corresponding `key` index (either in hexanumerical # format  or default python color string)
        alpha : float, optional
            transparency index (0 -> transparent; 1 -> opaque; default = 1.0)
        '''
        self.hashblocks[key] = [color,alpha]

    def gradmap(self,colormap=cm.cool,cmap_alpha=1):
        self.colormap = colormap
        self.cmap_alpha = cmap_alpha
        '''update colormap for gradient coloring

        Parameters
        ----------
        colormap : object, optional  
            colormap function from available Python colormaps (default cm.cool)
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
        alpha : float, optional
            transparency index (0 -> transparent; 1 -> opaque; default = 1.0)
        '''

    def build(self):
        binarray=binarize(self.array)   # `binarize` array 
        Z,X,Y = np.shape(self.array)

        z,x,y = np.indices((Z,X,Y))

        'cubes below max cubes(max values in array)'
        voxels = (x == 0) & (y == 0) & (z < binarray[0][0])

        for k in range(Z):
            for i in range(X):
                for j in range(Y):
                    if binarray[k,i,j]== 1:
                        cube0 = (x == i) & (y == j) & (z == k)

                        voxels = voxels | cube0

        return voxels
    
    def draw(self,coloring='adatoms'):
        Z,X,Y = np.shape(self.array)

        voxcolors=np.ones((Z,X,Y), dtype=object)

        def gradient_nuclear_coloring():

            colormap =  self.colormap
            alphaval =  self.cmap_alpha
            def center_color(a,L):
                return abs( 1 - 2*((a+0.5)/L))

            for i in range(X):
                for j in range(Y):
                    for k in range(Z):
                        ic,jc,kc = center_color(i,X), center_color(j,Y), center_color(k,Z)
                        voxcolors[k][i][j] = colormap(max(ic,jc,kc)-0.12,alpha =alphaval)


        def gradient_linear_coloring():
            colormap =  self.colormap
            alphaval =  self.cmap_alpha
            for i in range(X):
                for j in range(Y):
                    for k in range(Z):
                        voxcolors[k][i][j] = colormap((j*2+0.5)/Y-0.00, alpha=alphaval)



        def voxel_coloring():

            voxel_colmap = self.hashblocks
            # print(voxel_colmap)
            
            for i in range(X):
                for j in range(Y):
                    for k in range(Z):
                        if self.array[k][i][j] in voxel_colmap:
                            vxc = voxel_colmap[self.array[k][i][j]]
                            vxc_str, vxc_alpha = vxc
                            vxc_rgba = (*colors.to_rgb(vxc_str),vxc_alpha)
                            voxcolors[k][i][j] = vxc_rgba


        [gradient_nuclear_coloring() if coloring == 'nuclear' \
        else gradient_linear_coloring() if coloring == 'linear' \
        else voxel_coloring() if coloring == 'voxels' else None]

        fig = plt.figure()

        ax = fig.gca(projection='3d')
        ax._axis3don = False
        # ax.set_aspect('equal')         

        voxels = Model(self.array).build()

        ax.voxels(voxels, facecolors=voxcolors)

        set_axes_equal(ax)             # important!
        plt.show()