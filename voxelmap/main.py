'''
VOXELMAP 
A Python library for making voxel models from NumPy arrays.
Andrew Garcia, 2022

'''
import numpy as np
import random as ran
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib import colors

import pyvista 

from voxelmap.annex import *

def binarize(array,colorindex=1):
    '''converts an array with integer entries to either 0 if 0 or 1 if !=0'''
    arrayv = np.zeros((array.shape))
    for i in np.argwhere(array!=0):
        arrayv[tuple(i)] = colorindex
    
    return arrayv

class Model:

    def __init__(self,array=[]):
        '''Model structure. Calls `3-D` array to process into 3-D model.

        Parameters
        ----------
        array : np.array(int)
            array of the third-order populated with discrete, non-zero integers which may represent a voxel block type
        hashblocks : dict[int]{str, float }
            a dictionary for which the keys are the integer values on the discrete arrays (above) an the values are the color (str) for the specific key and the alpha for the voxel object (float)
        '''
        self.array = array          # array of third-order (3-D)
        self.hashblocks = {}        # start with empty voxel-color dictionary
        self.colormap = cm.cool     # default: cool colormap
        self.alphacm = 1            # default: opaque colormap (alpha=1)

    def hashblocksAdd(self, key, color, alpha=1):
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



    def build(self):
        '''Builds voxel model structure from python numpy array'''
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

    
    def draw_mpl(self,coloring='nuclear',edgecolors=None,figsize=(6.4,4.8),axis3don=False):
        ''' DRAW MATPLOTLIB.VOXELS
        Draws voxel model after building it with the provided `array`. 
        
        Parameters
        ----------
        coloring: string  
            voxel coloring scheme
                'nuclear'  colors model radially, from center to exterior
                'linear'   colors voxel model vertically, top to bottom. 
                'voxels'   colors voxel model based on the provided keys to its array integers, defined in the `hashblocks` variable from the `Model` class
        edgecolors: string/hex
            edge color of voxels (default: None)
        figsize : (float,float)
            defines plot window dimensions. From matplotlib.pyplot.figure(figsize) kwarg. 
        axis3don: bool
            defines presence of 3D axis in voxel model plot (Default: False)
        '''

        Z,X,Y = np.shape(self.array)

        voxcolors=np.ones((Z,X,Y), dtype=object)

        def gradient_nuclear_coloring():

            colormap =  self.colormap
            alphaval =  self.alphacm
            def center_color(a,L):
                return abs( 1 - 2*((a+0.5)/L))

            for i in range(X):
                for j in range(Y):
                    for k in range(Z):
                        ic,jc,kc = center_color(i,X), center_color(j,Y), center_color(k,Z)
                        voxcolors[k][i][j] = colormap(max(ic,jc,kc)-0.12,alpha =alphaval)


        def gradient_linear_coloring():
            colormap =  self.colormap
            alphaval =  self.alphacm
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

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(projection='3d')
        ax._axis3don = axis3don

        voxels = Model(self.array).build()

        ax.voxels(voxels, facecolors=voxcolors,edgecolors=edgecolors) 
        set_axes_equal(ax)           
        plt.show()

    def draw(self, coloring='none', scalars='', background_color = '#cccccc',window_size=[1024, 768]):
        '''Draws voxel model after building it with the provided `array` with PYVISTA
        
        Parameters
        ----------
        coloring: string  
            voxel coloring scheme
                'voxels' --> colors voxel model based on the provided keys to its array integers, defined in the `hashblocks` variable from the `Model` class
                'none'   --> no coloring
                ELSE:  coloring == cmap (colormap)
                'cool'      cool colormap
                'fire'      fire colormap
                and so on...
        scalars : list
            list of scalars for cmap coloring scheme
        background_color : string / hex
            background color of pyvista plot
        window_size : (float,float)
            defines plot window dimensions. Defaults to [1024, 768], unless set differently in the relevant theme’s window_size property [pyvista.Plotter]
        '''

        xx,yy,zz, voxid = arr2crds(self.array,-1).T

        centers = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

        pl = pyvista.Plotter(window_size=window_size)

        if background_color != "": 
            pl.background_color = background_color 

        for i in range(len(centers)):

            voxel = pyvista.Cube(center=centers[i]) 

            if coloring=='voxels':
                voxel_color, voxel_alpha = self.hashblocks[voxid[i]]
                pl.add_mesh(voxel, color=voxel_color, opacity=voxel_alpha)
            elif coloring=='none':
                pl.add_mesh(voxel)
            else:
                pl.add_mesh(voxel,scalars=[i for i in range(8)] if scalars=='' else scalars,cmap=coloring)


        pl.isometric_view_interactive()
        pl.show(interactive=True)



    def save(self,filename='voxeldata.json'): 
        '''Save sparse array + color assignments Model data as a dictionary of keys (DOK) JSON file
        
        Parameters
        ----------
        filename: string  
            name of file (e.g. 'voxeldata.json')
        '''
        tojson(filename, self.array, self.hashblocks)


    def load(self,filename='voxeldata.json'): 
        '''Load Model data represented as (DOK) JSON file to Model object
        
        Parameters
        ----------
        filename: string  
            name of file to be loaded (e.g 'voxeldata.json')
        '''
        data = load_from_json(filename)
        self.hashblocks = data['hashblocks']

        self.hashblocks  = {int(k):v for k,v in self.hashblocks .items()}
        
        Z,Y,X = data["size"]
        self.array = np.zeros((Z,Y,X))
        for c in range(len(data['coords'])):
            z,y,x = data['coords'][c]
            self.array[z,y,x] = data['val'][c]
