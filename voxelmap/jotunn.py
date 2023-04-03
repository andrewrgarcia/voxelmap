'''
VOXELMAP 
A Python library for making voxel models from NumPy arrays.
Andrew Garcia, 2022 - beyond

'''
import numpy as np
import random as ran
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib import colors
import pandas

import pyvista

import cv2
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from scipy.spatial import ConvexHull

from voxelmap.annex import *

# from scipy.spatial import Delaunay



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

        if plot=='mpl': 
            for s in hull.simplices:
                s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                ax.plot(Y_here+points[s, 1],X_here+points[s, 0], Z_here+points[s, 2], color=color)

        newsimplices = np.array([s + num_simplices for s in hull.simplices])


        newpoints = np.array([ i+[Y_here,X_here,Z_here] for i in points]) if sector_dims == 2 else \
                    np.array([ i+[Z_here,Y_here,X_here] for i in points]) 


        return newpoints, newsimplices

    else:
        return np.empty(0), np.empty(0)



def binarize(array, colorindex=1):
    '''converts an array with integer entries to either 0 if 0 or 1 if !=0'''
    arrayv = np.zeros((array.shape))
    for i in np.argwhere(array != 0):
        arrayv[tuple(i)] = colorindex

    return arrayv


class Model:

    def __init__(self, array=[],file=''):
        '''Model structure. Calls `3-D` array to process into 3-D model.

        Parameters
        ----------
        array : np.array(int)
            array of the third-order populated with discrete, non-zero integers which may represent a voxel block type
        file : str
            file name and/or path for image file
        hashblocks : dict[int]{str, float }
            a dictionary for which the keys are the integer values on the discrete arrays (above) an the values are the color (str) for the specific key and the alpha for the voxel object (float)
        colormap : < matplotlib.cm object >
            colormap to use for voxel coloring if coloring kwarg in Model.draw method is not voxels. Default: cm.cool
        alphacm : float
            alpha transparency of voxels if colormap option is chosen. default: opaque colormap (alpha=1)
        
        -- FOR FILE PROCESSING --

        file : str
            file name a/or path for goxel txt file

        -- FOR XYZ COORDINATE ARRAY PROCESSING -- 

        XYZ : np.array(float )
            an array containing the x,y,z coordinates of shape `number_voxel-locations` x 3 [for each x y z]
        RGB : list[str] 
            a list for the colors of every voxels in xyz array (length: `number_voxel-locations`)
        sparsity : float
            a factor to separate the relative distance between each voxel (default:10.0 [> 50.0 may have memory limitations])
        '''
        self.file = file
        self.array = mpimg.imread(self.file) if file != '' else array   # array of third-order (3-D)
        self.make_intensity() if file != '' else None

        self.hashblocks = {}        # start with empty voxel-color dictionary
        self.colormap = cm.cool     # default: cool colormap
        self.alphacm = 1            # default: opaque colormap (alpha=1)

        # self.file = 'placeholder.txt'
        self.objfile = 'model.obj'
        self.XYZ = []
        self.RGB = []
        self.sparsity = 10.0
        

    def resize_intensity(self, res = 1.0, res_interp = cv2.INTER_AREA):
        '''Resize the intensity matrix of the provided image.

        Parameters
        ----------
        res : float, optional
            relative resizing percentage as `x` times the original (default 1.0 [1.0x original dimensions])
        res_interp: object, optional 
            cv2 interpolation function for resizing (default cv2.INTER_AREA)
        '''
        
        'Turn image into intensity matrix'
        # self.array = mpimg.imread(self.file)       #load image
        'Use CCIR 601 luma to convert RGB image to rel. grayscale 2-D matrix (https://en.wikipedia.org/wiki/Luma_(video)) '
        color_weights = [0.299,0.587,0.114]
        intensity = np.sum([self.array[:,:,i]*color_weights[i] for i in range(3)],0)*100

        'resize_intensity'
        x,y = intensity.shape
        x,y = int(x*res),int(y*res)
        intensity = cv2.resize(intensity, (x,y), interpolation = res_interp )

        self.array = intensity.astype('float')  # floor-divide


    def make_intensity(self):
        '''Turn image into intensity matrix i.e. matrix with pixel intensities. Outputs self.array as mutable matrix to contain relative pixel intensities in int levels [for file if specified]
        '''
        # if self.array == []:
        #     self.array = mpimg.imread(self.file)       #load image
        
        'Use CCIR 601 luma to convert RGB image to rel. grayscale 2-D matrix (https://en.wikipedia.org/wiki/Luma_(video)) '
        color_weights = [0.299,0.587,0.114]
        intensity = np.sum([self.array[:,:,i]*color_weights[i] for i in range(3)],0)*100

        # 'if resize'
        # if res != 1.0:
        #     x,y = intensity.shape
        #     x,y = int(x*res),int(y*res)
        #     self.array = cv2.resize(intensity, (x,y), interpolation = res_interp )

        self.array = intensity.astype('float')  # floor-divide


    def importdata(self, filename=''):

        def dfgox(file):
            '''Import Goxel file and convert to numpy array
            '''
            df = pandas.read_csv(file,
                                 sep=' ', skiprows=(0, 1, 2), names=['x', 'y', 'z', 'rgb', 'none'])

            return df

        def dfXYZ():
            '''Import Goxel file and convert to numpy array
            '''
            df = pandas.DataFrame(self.sparsity*self.XYZ,
                                  columns=['x', 'y', 'z'])

            df['rgb'] = self.RGB if len(self.RGB) else len(df)*['a3ebb1']

            return df

        df = dfgox(filename) if len(filename) != 0 else dfXYZ()

        minx, miny, minz = df.min()[0:3]
        maxx, maxy, maxz = df.max()[0:3]

        df['z'] += (-minz)
        df['y'] += (-miny)
        df['x'] += (-minx)

        Z, Y, X = int(maxz-minz+1), int(maxy-miny+1), int(maxx-minx+1)

        self.array = np.zeros((Z, Y, X))

        elems = df.T

        'define voxel hashblocks dict from colors present in voxel file'
        model_colors = sorted(list(set(df['rgb'])))

        if isinstance(model_colors[0], str):
            for i in range(len(model_colors)):
                # self.hashblocks.update({i+1: model_colors[i] })
                self.hashblocks[i+1] = ['#'+model_colors[i], 1]
            print('Color list built from file!\nself.hashblocks =\n',self.hashblocks)

        'write array from .txt file voxel color values and locs'
        for i in range(len(elems.T)):
            x, y, z = elems[i][0:3].astype('int')

            if isinstance(model_colors[0], str):
                rgb = '#'+elems[i][3]
                self.array[z, y, x] = [
                    i for i in self.hashblocks if self.hashblocks[i][0] == rgb][0]
            else:
                self.array[z, y, x] = elems[i][3]

        return self.array

    def hashblocks_add(self, key, color, alpha=1):
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
        self.hashblocks[key] = [color, alpha]

    def build(self):
        '''Builds voxel model structure from python numpy array'''
        binarray = binarize(self.array)   # `binarize` array
        Z, X, Y = np.shape(self.array)

        z, x, y = np.indices((Z, X, Y))

        'cubes below max cubes(max values in array)'
        voxels = (x == 0) & (y == 0) & (z < binarray[0][0])

        for k in range(Z):
            for i in range(X):
                for j in range(Y):
                    if binarray[k, i, j] == 1:
                        cube0 = (x == i) & (y == j) & (z == k)

                        voxels = voxels | cube0

        return voxels

    def draw_mpl(self, coloring='custom', edgecolors=None, figsize=(6.4, 4.8), axis3don=False):
        """
        Draws voxel model after building it with the provided `array` (Matplotlib version. For faster graphics, try the ``draw()`` method (uses PyVista)). 

        Parameters
        ----------
        coloring: string  
            voxel coloring scheme
                * 'custom' --> colors voxel model based on the provided keys to its array integers, defined in the `hashblocks` variable from the `Model` class
                * 'custom: #8599A6' -->  color all voxel types with the #8599A6 hex color (bluish dark gray) and an alpha transparency of 1.0 (default)
                * 'custom: red, alpha: 0.24' --> color all voxel types red and with an alpha transparency of 0.2
                * 'nuclear'  colors model radially, from center to exterior
                * 'linear'   colors voxel model vertically, top to bottom. 
        edgecolors: string/hex
            edge color of voxels (default: None)
        figsize : (float,float)
            defines plot window dimensions. From matplotlib.pyplot.figure(figsize) kwarg. 
        axis3don: bool
            defines presence of 3D axis in voxel model plot (Default: False)
        """

        Z, X, Y = np.shape(self.array)

        voxcolors = np.ones((Z, X, Y), dtype=object)

        def gradient_nuclear_coloring():

            colormap = self.colormap
            alphaval = self.alphacm

            def center_color(a, L):
                return abs(1 - 2*((a+0.5)/L))

            for i in range(X):
                for j in range(Y):
                    for k in range(Z):
                        ic, jc, kc = center_color(i, X), center_color(
                            j, Y), center_color(k, Z)
                        voxcolors[k][i][j] = colormap(
                            max(ic, jc, kc)-0.12, alpha=alphaval)

        def gradient_linear_coloring():
            colormap = self.colormap
            alphaval = self.alphacm
            for i in range(X):
                for j in range(Y):
                    for k in range(Z):
                        voxcolors[k][i][j] = colormap(
                            (j*2+0.5)/Y-0.00, alpha=alphaval)

        def voxel_coloring():

            color_details= coloring.split(':')

            if len(color_details) > 1:
                if len(color_details) > 2:
                    color_all = color_details[1].split(',')[0].strip()
                    alpha_all  = float(color_details[2])
                else: 
                    color_all = color_details[1].strip()
                    alpha_all  = 1.0

                iterlist = np.unique(self.array[self.array!=0]) if len(self.hashblocks)==0 else self.hashblocks.keys()      #iterate list over all non-zero integer types 

                for i in iterlist:
                    self.hashblocks[i] = [color_all,alpha_all] 
                    
            print('Voxelmap draw. Using custom colors:\nself.hashblocks =\n',self.hashblocks)

            voxel_colmap = self.hashblocks
            # print(voxel_colmap)

            for i in range(X):
                for j in range(Y):
                    for k in range(Z):
                        if self.array[k][i][j] in voxel_colmap:
                            vxc = voxel_colmap[self.array[k][i][j]]
                            vxc_str, vxc_alpha = vxc
                            vxc_rgba = (*colors.to_rgb(vxc_str), vxc_alpha)
                            voxcolors[k][i][j] = vxc_rgba

        [gradient_nuclear_coloring() if coloring == 'nuclear'
        else gradient_linear_coloring() if coloring == 'linear'
        else voxel_coloring() if coloring[:6] == 'custom' else None]

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(projection='3d')
        ax._axis3don = axis3don

        voxels = Model(self.array).build()

        ax.voxels(voxels, facecolors=voxcolors, edgecolors=edgecolors)
        set_axes_equal(ax)
        plt.show()

    def draw(self, coloring='none', geometry = 'voxels', scalars='', background_color='#cccccc', wireframe=False, window_size=[1024, 768],voxel_spacing=(1,1,1)):
        '''Draws voxel model after building it with the provided `array` with PyVista library

        Parameters
        ----------
        coloring: string  
            voxel coloring scheme
                * 'custom' --> colors voxel model based on the provided keys to its array integers, defined in the `hashblocks` variable from the `Model` class
                * 'custom: #8599A6' -->  color all voxel types with the #8599A6 hex color (bluish dark gray) and an alpha transparency of 1.0 (default)
                * 'custom: red, alpha: 0.24' --> color all voxel types red and with an alpha transparency of 0.24
                * 'none'   --> no coloring 
                * 'cool'      cool colormap
                * 'fire'      fire colormap
                * and so on...
        geometry: string  
            voxel geometry. Choose voxels to have a box geometry with geometry='voxels' or spherical one with geometry='particles'
        scalars : list
            list of scalars for cmap coloring scheme
        background_color : string / hex
            background color of pyvista plot
        window_size : (float,float)
            defines plot window dimensions. Defaults to [1024, 768], unless set differently in the relevant theme’s window_size property [pyvista.Plotter]
        voxel_spacing : (float,float,float)
            changes voxel spacing by defining length scales of x y and z directions (default:(1,1,1)).
        '''

        xx, yy, zz, voxid = arr2crds(self.array, -1).T

        centers = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

        pl = pyvista.Plotter(window_size=window_size)

        if background_color != "":
            pl.background_color = background_color

        if coloring[:6] == 'custom':
            color_details= coloring.split(':')

            if len(color_details) > 1:
                if len(color_details) > 2:
                    color_all = color_details[1].split(',')[0].strip()
                    alpha_all  = float(color_details[2])
                else: 
                    color_all = color_details[1].strip()
                    alpha_all  = 1.0

                iterlist = np.unique(self.array[self.array!=0]) if len(self.hashblocks)==0 else self.hashblocks.keys()      #iterate list over all non-zero integer types 

                for i in iterlist:
                    self.hashblocks[i] = [color_all,alpha_all] 
                        
            print('Voxelmap draw. Using custom colors:\nself.hashblocks =\n',self.hashblocks)

        for i in range(len(centers)):

            x_len,y_len,z_len = voxel_spacing

            if geometry == 'particles':
                voxel = pyvista.Sphere(center=centers[i],radius=0.5)
                smooth = True
            else:
                voxel = pyvista.Cube(center=centers[i],x_length=x_len, y_length=y_len, z_length=z_len)
                smooth= None

            if coloring[:6] == 'custom' :

                voxel_color, voxel_alpha = self.hashblocks[voxid[i]]
                pl.add_mesh(voxel, color=voxel_color, smooth_shading=smooth, opacity=voxel_alpha,show_edges=True if wireframe else False)
            elif coloring == 'none':
                pl.add_mesh(voxel,smooth_shading=smooth, show_edges=True if wireframe else False)
            else:
                pl.add_mesh(voxel, scalars=[i for i in range(
                    8)] if scalars == '' else scalars,smooth_shading=smooth, show_edges=True if wireframe else False, cmap=coloring)

        pl.isometric_view_interactive()
        pl.show(interactive=True)

    def save(self, filename='voxeldata.json'):
        '''Save sparse array + color assignments Model data as a dictionary of keys (DOK) JSON file

        Parameters
        ----------
        filename: string  
            name of file (e.g. 'voxeldata.json')
            Data types:
            .json -> voxel data represented as (DOK) JSON file 
            .txt -> voxel data represented x,y,z,rgb matrix in .txt file (see Goxel .txt imports)
        '''
        if filename[-4:] == 'json':
            tojson(filename, self.array, self.hashblocks)
        else:
            toTXT(filename,self.array, self.hashblocks)

        return None
    
    def load(self, filename='voxeldata.json', coords=False):
        """
        Load to Model object.

        Parameters
        ----------
        filename: string (.json or .txt extensions (see above))
            name of file to be loaded (e.g 'voxeldata.json')
        coords: bool
            loads and processes self.XYZ, self.RGB, and self.sparsity = 10.0 (see Model class desc above) to Model if True. This boolean overrides filename loader option. 
        """
        if coords:
            self.importdata(filename='')
        else:
            if filename[-4:] == 'json':

                data = load_from_json(filename)
                self.hashblocks = data['hashblocks']

                self.hashblocks = {
                    int(k): v for k, v in self.hashblocks .items()}

                Z, Y, X = data["size"]
                self.array = np.zeros((Z, Y, X))
                for c in range(len(data['coords'])):
                    z, y, x = data['coords'][c]
                    self.array[z, y, x] = data['val'][c]

            else:
                self.importdata(filename)



    def ImageMap(self,depth=5,out_file='model.obj',plot = False):
        '''Map image or 2-D array (matrix) to 3-D array
        
        Parameters
        -----------
        depth : int
            depth in number of voxels (default = 5 voxels)
        out_file : str
            name and/or path for Wavefront .obj file output. This is the common format for OpenGL 3-D model files (default: model.obj) 
        plot: bool / str
            plots a preliminary 3-D triangulated image if True with PyVista. For more plotting options, plot by calling MeshView separately.
        '''
        matrix = self.array

        length, width = np.shape(matrix)

        low, high = np.min(matrix),np.max(matrix)
        intensities = np.linspace(low,high,depth).astype('int')
        
        model = np.zeros((depth,length,width))      # model (3D array)


        for j in range(length):
            for i in range(width):
                pixel_intensity = matrix[j,i]
                # k = findcrossover(intensities,0,depth-1, pixel_intensity)
                # model[k-1][j][ i ] = 1
                k = findclosest(intensities, pixel_intensity)
                model[k][j][ i ] = 1

        voxelwrite(model, filename =out_file)
        self.objfile = out_file 

        if plot:
            self.MeshView()

        return model
    
    def ImageMesh(self, out_file='model.obj', L_sectors = 4, rel_depth = 0.50, trace_min = 1, plot = True, figsize=(4.8,4.8), verbose=False ):
        '''
        3-D triangulation of 2-D images / 2-D arrays (matrices) with a Convex Hull algorithm (Andrew Garcia, 2022)

        Parameters
        ------------
        out_file : str
            name and/or path for Wavefront .obj file output. This is the common format for OpenGL 3-D model files (default: model.obj) 
        L_sectors: int
            length scale of Convex Hull segments in sector grid, e.g. L_sectors = 4 makes a triangulation of 4 x 4 Convex Hull segments
        rel_depth: float
            relative depth of 3-D model with respect to the image's intensity magnitudes (default: 0.50)
        trace_min: int
            minimum number of points in different z-levels to triangulate per sector (default: 1)
        plot: bool / str
            plots a preliminary 3-D triangulated image if True [with PyVista (& with matplotlib if plot = 'img'). For more plotting options, plot with Meshview instead. 
        '''

        matrix = self.array

        print(matrix)
        L = matrix.shape[0]
        W = matrix.shape[1]
        print(L,W)

        ax=[]

        if plot=='mpl':
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
            if plot == 'mpl':
                ax.set_title("{} Convex Hull segments".format(L_sectors**2),color="#D3D3D3")
                ax.set_facecolor('#3e404e')

                set_axes_equal(ax)
                plt.axis('off')
                plt.show()
            else:
                self.MeshView()


    def MarchingMesh(self, voxel_depth=12, level=0, spacing=(1., 1., 1.), gradient_direction='descent', step_size=1, allow_degenerate=True, method='lewiner', mask=None,plot=False, figsize=(4.8,4.8) ):
        """
        Marching cubes on 3-D mapped image or 3-D array

        Parameters
        ---------------
        voxel_depth : int (optional)
            depth of 3-D mapped image on number of voxels (if file [image] is to be processed / i.e. specified)
        level : float, optional
            Contour value to search for isosurfaces in `volume`. If not given or None, the average of the min and max of vol is used.
        spacing : length-3 tuple of floats, optional
            Voxel spacing in spatial dimensions corresponding to numpy array indexing dimensions (M, N, P) as in `volume`.
        gradient_direction : string, optional
            Controls if the mesh was generated from an isosurface with gradient descent toward objects of interest (the default), or the opposite, considering the *left-hand* rule.
            The two options are: -- 'descent' : Object was greater than exterior -- 'ascent' : Exterior was greater than object
        step_size : int, optional
            Step size in voxels. Default 1. Larger steps yield faster but coarser results. The result will always be topologically correct though.
        allow_degenerate : bool, optional
            Whether to allow degenerate (i.e. zero-area) triangles in the end-result. Default True. If False, degenerate triangles are removed, at the cost of making the algorithm slower.
        method: str, optional
            One of 'lewiner', 'lorensen' or '_lorensen'. Specify which of Lewiner et al. or Lorensen et al. method will be used. The '_lorensen' flag correspond to an old implementation that will be deprecated in version 0.19.
        mask : (M, N, P) array, optional
            Boolean array. The marching cube algorithm will be computed only on True elements. This will save computational time when interfaces are located within certain region of the volume M, N, P-e.g. the top half of the cube-and also allow to compute finite surfaces-i.e. open surfaces that do not end at the border of the cube.
        plot: bool
            plots a preliminary 3-D triangulated image if True
        """

        # image in self.file mapped to 3d if self.file image specified else it takes the 3-D array defined in self.array
        array = self.ImageMap(voxel_depth) if self.file != '' else self.array              

        MarchingMesh(array, out_file=self.objfile, level=level,spacing=spacing, gradient_direction=gradient_direction, step_size=step_size, allow_degenerate=allow_degenerate, method=method, mask=mask, plot=plot, figsize=figsize)
        
        print('mesh created! saved as {}.'.format(self.objfile))

    def MeshView(self,wireframe=False,color='pink',alpha=0.5,background_color='#333333', viewport = [1024, 768]):
        """
        Triangulated mesh view with PyVista

        Parameters
        --------------
        objfile: string
            .obj file to process with MeshView [in GLOBAL function only]
        wireframe: bool
            Represent mesh as wireframe instead of solid polyhedron if True (default: False). 
        color : string / hexadecimal
            mesh color. default: 'pink'
        alpha : float
            opacity transparency range: 0 - 1.0. Default: 0.5
        background_color : string / hexadecimal
            color of background. default: 'pink'
        viewport : (int,int)
            viewport / screen (width, height) for display window (default: 80% your screen's width & height)
        """
        mesh = pv.read(self.objfile)
        mesh.plot(show_edges=True if wireframe else False, color=color,opacity=alpha,background=background_color,window_size = viewport)