
import matplotlib.image as mpimg
import numpy as np
import cv2
import pygame


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
# from scipy.spatial import Delaunay

from voxelmap.annex import *
import voxelmap.objviewer as viewer

from scipy import ndimage 


def resize_array(array, mult=(2,2,2), threshold=1):
    '''Resizes a three-dimensional array by the three dim factors specified by `mult` tuple. 
    Converts to sparse array of 0s and 1s   

    Parameters
    ----------
    array : np.array(int,int,int)
        array to resize
    mult: tuple(float,float,float)
        depth length width factors to resize array with. e.g 2,2,2 resizes array to double its size in all dims
    threshold : float / iint
        Threshold is minimum value to convert to 1 in binary array. 
    '''
    array = ndimage.zoom(array, (2,2,2))
    crds_nonzero = np.argwhere(array > threshold)
    array.fill(0)
    for k in crds_nonzero:
        array[tuple(k)] = 1

    return array


def random_kernel_convolve(array, kernel,random_bounds=(-10,10)):
    '''Applies a three-dimensional convolution with a randomly-mutating `kernel` 
    on a 3-D `array` which changes for every array site when random_bounds are set to tuple. 
    If random_bounds are set to False, convolution occurs in constant mode for the specified kernel. 

    Parameters
    ----------
    array : np.array(int,int,int)
        array to convolve
    kernel: np.array(int,int,int)
        kernel to use for convolution. If random_bounds are set to tuple, only the kernel's shape is used to specify the random_kernels
    random_bounds : tuple(int,int) OR bool
        see above explanation.
    '''
    if random_bounds and kernel.shape[0]%2:	
        new_array = np.zeros(array.shape)

        array = np.pad(array,kernel.shape[0]//2)

        k_z,k_y,k_x = kernel.shape
        Z,Y,X = new_array.shape 
        # adapted from: 
        # https://towardsdatascience.com/tensorflow-for-computer-vision-how-to-implement-convolutions-from-scratch-in-python-609158c24f82
        for k in range(Z):
            for j in range(Y):
                for i in range(X):
                    sector = array[k:k+k_z, j:j+k_y, i:i+k_x]

                    kernel = np.random.randint(*random_bounds,(k_z,k_y,k_x)) 
                    new_array[k,j,i] = np.sum(np.multiply(sector, kernel))
        return new_array

    else:
        return ndimage.convolve(array, kernel, mode='constant', cval=0.0)

def roughen(array,kernel_level=1):
    '''Makes a 3d array model rougher by a special convolution operation. Uses `voxelmap.random_kernel_convolve`.

    Parameters
    ----------
    array : np.array(int,int,int)
        array to `roughen up`
    kernel_level: int
        length scale (size) of random kernels used. The smallest scale (=1) gives the roughest transformation.
    '''
    kernel = np.zeros((kernel_level,kernel_level,kernel_level))
    return random_kernel_convolve(array,kernel,(-1,2))

def SectorHull(array, sector_dims, Z_here, Z_there, Y_here, Y_there, X_here, X_there,  
                num_simplices, rel_depth, color='orange', trace_min=1, plot=True, ax=[]):
    '''SectorHull does ConvexHull on a specific 2-D sector of the selected image
    adapted [ significantly ] from 
    https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud'''

    sector = array[Y_here:Y_there, X_here:X_there] if sector_dims == 2 else \
             array[Z_here:Z_there, Y_here:Y_there, X_here:X_there]


    if len(np.unique(sector)) > trace_min:
        
        points = arr2crds(sector, rel_depth) if sector_dims == 2 else tensor2crds(array, rel_depth)

        hull = ConvexHull(points)

        if plot: 
            for s in hull.simplices:
                s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                ax.plot(Y_here+points[s, 1],X_here+points[s, 0], Z_here+points[s, 2], color=color)

        newsimplices = np.array([s + num_simplices for s in hull.simplices])


        newpoints = np.array([ i+[Y_here,X_here,Z_here] for i in points]) if sector_dims == 2 else \
                    np.array([ i+[Z_here,Y_here,X_here] for i in points]) 


        return newpoints, newsimplices

    else:
        return np.empty(0), np.empty(0)


class Image:
    def __init__(self,file):
        '''Image structure.

        Parameters
        ----------
        file : str
            file name and/or path for image file
        intensity : array((int,int)))
            mutable matrix to contain relative pixel intensities in int levels \
        tensor: array((int,int,int)))
            third-order tensor to represent voxel / point-cloud models in array form 
        objfile: str
            name of Wavefront (.obj) file to make from ImageMesh
        '''
        self.file = file
        self.array = []
        self.intensity = np.ones((5,5))
        self.tensor = np.ones((5,5,5))
        self.objfile = 'model.obj'

    def resize(self, res = 1.0, res_interp = cv2.INTER_AREA):
        '''Resize the intensity matrix of the provided image.

        Parameters
        ----------
        res : float, optional
            relative resizing percentage as `x` times the original (default 1.0 [1.0x original dimensions])
        res_interp: object, optional 
            cv2 interpolation function for resizing (default cv2.INTER_AREA)
        '''
        
        'Turn image into intensity matrix'
        self.array = mpimg.imread(self.file)       #load image
        'Use CCIR 601 luma to convert RGB image to rel. grayscale 2-D matrix (https://en.wikipedia.org/wiki/Luma_(video)) '
        color_weights = [0.299,0.587,0.114]
        self.intensity = np.sum([self.array[:,:,i]*color_weights[i] for i in range(3)],0)*100

        'resize'
        x,y = self.intensity.shape
        x,y = int(x*res),int(y*res)
        self.intensity = cv2.resize(self.intensity, (x,y), interpolation = res_interp )

        self.intensity = self.intensity.astype('int')  # floor-divide


    def make(self):
        '''Turn image into intensity matrix i.e. matrix with pixel intensities
        '''
        if self.array == []:

            self.array = mpimg.imread(self.file)       #load image
            'Use CCIR 601 luma to convert RGB image to rel. grayscale 2-D matrix (https://en.wikipedia.org/wiki/Luma_(video)) '
            color_weights = [0.299,0.587,0.114]
            self.intensity = np.sum([self.array[:,:,i]*color_weights[i] for i in range(3)],0)*100

        # 'if resize'
        # if res != 1.0:
        #     x,y = self.intensity.shape
        #     x,y = int(x*res),int(y*res)
        #     self.intensity = cv2.resize(self.intensity, (x,y), interpolation = res_interp )

        self.intensity = self.intensity.astype('int')  # floor-divide

    def ImageMap(self,depth=5):
        '''Map image to 3-D array 

        Parameters
        ----------
        depth : int
            depth in number of voxels (default = 5 voxels)
        '''
        intensity = self.intensity

        length, width = np.shape(intensity)

        low, high = np.min(intensity),np.max(intensity)
        intensities = np.linspace(low,high,depth).astype('int')
        
        model = np.zeros((depth,length,width))      # model (3D array)


        for j in range(length):
            for i in range(width):
                pixel_intensity = intensity[j,i]
                # k = findcrossover(intensities,0,depth-1, pixel_intensity)
                # model[k-1][j][ i ] = 1
                k = findclosest(intensities, pixel_intensity)
                model[k][j][ i ] = 1

        return model

    def ImageMesh(self, out_file='model.obj', L_sectors = 4, rel_depth = 0.50, trace_min = 5,\
                        plot = True, figsize=(4.8,4.8), verbose=False ):
        '''3-D triangulation of 2-D images with a Convex Hull algorithm
        Andrew Garcia, 2022

        Parameters
        ----------
        out_file : str
            name and/or path for Wavefront .obj file output. This is the common format for OpenGL 3-D model files (default: model.obj) 
        L_sectors: int
            length scale of Convex Hull segments in sector grid, e.g. L_sectors = 4 makes a triangulation of 4 x 4 Convex Hull segments
        rel_depth: float
            relative depth of 3-D model with respect to the image's intensity magnitudes (default: 0.50)
        trace_min: int
            minimum number of points in different z-levels to triangulate per sector (default: 5)
        plot: bool
            plots a preliminary 3-D triangulated image if True
        '''

        matrix = self.intensity

        L = matrix.shape[0]
        W = matrix.shape[1]

        ax=[]
        if plot:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        "multiple sectors"
        NUM = 0
        ls = np.linspace(0,1,L_sectors+1)
        ws = np.linspace(0,1,L_sectors+1)
        k=0
        for i in range(L_sectors):
            for j in range(L_sectors):

                newpts, newsmpls = SectorHull(matrix, 2, 0, 0, int(ls[i]*L),int(ls[i+1]*L),\
                                    int(ws[j]*W),int(ws[j+1]*W),NUM,rel_depth,'lime',trace_min, plot,ax)
                NUM+=len(newpts)
                if verbose:
                    print(NUM)
                    print(newpts.shape, newsmpls.shape)
                if newpts.shape[0]:
                    points2 = np.concatenate((points2,newpts)) if k!=0 else newpts
                    hullsimplices2 = np.concatenate((hullsimplices2,newsmpls)) if k!=0 else newsmpls
                    k+=1

        if verbose:    
            print('points shape',points2.shape)
            print('simplices shape',hullsimplices2.shape)

        points2n=points2*5/np.max(points2)
        writeobj_CH(points2n,hullsimplices2,out_file)

        self.objfile = out_file 

        if plot:
            ax.set_title("{} Convex Hull segments".format(L_sectors**2),color="#D3D3D3")
            ax.set_facecolor('#3e404e')

            set_axes_equal(ax)
            plt.axis('off')
            plt.show()

    def MarchingMesh(self, voxel_depth=12, level=0,
                spacing=(1., 1., 1.), gradient_direction='descent', step_size=1, 
                allow_degenerate=True, method='lewiner', mask=None,
                plot=False, figsize=(4.8,4.8) ):
        '''Marching cubes on 3D-mapped image

        Parameters
        ----------
        voxel_depth : int
            depth of 3-D mapped image on number of voxels

        --- FROM SKIMAGE.MEASURE.MARCHING_CUBES ---
        level : float, optional
            Contour value to search for isosurfaces in `volume`. If not
            given or None, the average of the min and max of vol is used.
        spacing : length-3 tuple of floats, optional
            Voxel spacing in spatial dimensions corresponding to numpy array
            indexing dimensions (M, N, P) as in `volume`.
        gradient_direction : string, optional
            Controls if the mesh was generated from an isosurface with gradient
            descent toward objects of interest (the default), or the opposite,
            considering the *left-hand* rule.
            The two options are:
            * descent : Object was greater than exterior
            * ascent : Exterior was greater than object

        step_size : int, optional
            Step size in voxels. Default 1. Larger steps yield faster but
            coarser results. The result will always be topologically correct
            though.
        allow_degenerate : bool, optional
            Whether to allow degenerate (i.e. zero-area) triangles in the
            end-result. Default True. If False, degenerate triangles are
            removed, at the cost of making the algorithm slower.
        method: str, optional
            One of 'lewiner', 'lorensen' or '_lorensen'. Specify which of
            Lewiner et al. or Lorensen et al. method will be used. The
            '_lorensen' flag correspond to an old implementation that will
            be deprecated in version 0.19.
        mask : (M, N, P) array, optional
            Boolean array. The marching cube algorithm will be computed only on
            True elements. This will save computational time when interfaces
            are located within certain region of the volume M, N, P-e.g. the top
            half of the cube-and also allow to compute finite surfaces-i.e. open
            surfaces that do not end at the border of the cube.

        plot: bool
            plots a preliminary 3-D triangulated image if True
        '''

        self.make()                             # resized to 1.0x original size i.e. not resized (default)
        mapped_img = self.ImageMap(voxel_depth)              # mapped to 3d with a depth of 12 voxels

        MarchingMesh(mapped_img, out_file=self.objfile, level=level,
                spacing=spacing, gradient_direction=gradient_direction, step_size=step_size, 
                allow_degenerate=allow_degenerate, method=method, mask=mask,
                plot=plot, figsize=figsize)


    # viewport_default = (0.8*np.array(pygame.display.set_mode().get_rect()[2:]))
    def MeshView(self, wireframe=False, viewport=(2048, 1152)):
        '''MeshView: triangulated mesh view with OpenGL [ uses pygame ]

        Parameters
        ----------
        wireframe: bool
            Represent mesh as wireframe instead of solid polyhedron if True (default: False). 
        viewport : (int,int)
            viewport / screen (width, height) for display window (default: 80% your screen's width & height)
        '''
        viewer.objview(self.objfile, wireframe=wireframe, usemtl=False, viewport=viewport)