
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

class Image:
    def __init__(self,file):
        '''Image structure.

        Parameters
        ----------
        file : str
            file name and/or path for image file
        intensity : array(float)
            mutable matrix to contain relative pixel intensities in int levels 
        '''
        self.file = file
        self.intensity = np.ones((5,5))
        self.objfile = 'model.obj'

    def make(self, res = 1.0, res_interp = cv2.INTER_AREA):
        '''Turn image into intensity matrix i.e. matrix with pixel intensities

        Parameters
        ----------
        res : float, optional
            relative resizing percentage as `x` times the original (default 1.0 [1.0x original dimensions])
        res_interp: object, optional 
            cv2 interpolation function for resizing (default cv2.INTER_AREA)
        '''
        img = mpimg.imread(self.file)       #load image
        'Use CCIR 601 luma to convert RGB image to rel. grayscale 2-D matrix (https://en.wikipedia.org/wiki/Luma_(video)) '
        color_weights = [0.299,0.587,0.114]
        self.intensity = np.sum([img[:,:,i]*color_weights[i] for i in range(3)],0)*100

        if res != 1.0:
            x,y = self.intensity.shape
            x,y = int(x*res),int(y*res)
            self.intensity = cv2.resize(self.intensity, (x,y), interpolation = res_interp )


        self.intensity = self.intensity.astype('int')  # floor-divide

        # print(mat2crds(self.intensity))
        # print(np.max(self.intensity))

    def map3d(self,depth=5):
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

    def ImageMesh(self, out_file='model.obj', plot = True, L_sectors = 4, rel_depth = 0.50, trace_min = 5 ):
        '''3-D triangulation of 2-D images with a Convex Hull algorithm
        Andrew Garcia, 2022

        Parameters
        ----------
        out_file : str
            name and/or path for Wavefront .obj file output. This is the common format for OpenGL 3-D model files (default: model.obj) 
        plot: bool
            plots a preliminary 3-D triangulated image if True
        L_sectors: int
            length scale of Convex Hull segments in sector grid, e.g. L_sectors = 4 makes a triangulation of 4 x 4 Convex Hull segments
        rel_depth: float
            relative depth of 3-D model with respect to the image's intensity magnitudes (default: 0.50)
        trace_min: int
            minimum number of points in different z-levels to triangulate per sector (default: 5)
        '''


        def writeobj(points, hull_simplices, filename = 'this.obj'):
            '''Writes the triangulated image, which makes a 3-D mesh model, as an .obj file.'''
            with open(filename, 'w') as f:
                for i in points:
                    f.write("v  {:.4f} {:.4f} {:.4f}\n".format(*i))


                block = """
    vt 1.00 0.00 0.00
    vt 1.00 1.00 0.00
    vt 0.00 1.00 0.00
    vt 0.00 0.00 0.00

    vn 0.00 0.00 -1.00
    vn 0.00 0.00 1.00
    vn 0.00 -1.00 0.00
    vn 1.00 0.00 0.00
    vn 0.00 1.00 0.00
    vn -1.00 0.00 0.00

    \n"""

                f.write("\n"+block)

                f.write("\ng Polyhedral\n\n")

                for j in hull_simplices:
                    # the vertex texture (vt) triangle indices which color a specific simplex are [ currently ] being defined at random 
                    rand_t0 = np.random.randint(4)
                    rand_n = np.random.randint(1,7)


                    j+=1    # hull simplices start at index 1 not 0 (this makes the correction)
                    j1,j2,j3 = j

                    facestr = [j1,(rand_t0+0)%4+1,rand_n,\
                            j2,(rand_t0+1)%4+1,rand_n,\
                            j3,(rand_t0+2)%4+1,rand_n  
                            ]

                    f.write("f {}/{}/{} {}/{}/{} {}/{}/{}\n".format(*facestr))


        def SectorHull(array, X_here, X_there, 
                      Y_here, Y_there,
                      simplices0, rel_depth, color='orange'):
            '''SectorHull does ConvexHull on a specific 2-D sector of the selected image
            adapted [ significantly ] from 
            https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud'''

        
            sector = array[X_here:X_there,Y_here:Y_there]

            if len(np.unique(sector)) > trace_min:
                points = arr2crds(sector,rel_depth)

                hull = ConvexHull(points)


                if plot:
                    for s in hull.simplices:
                        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                        ax.plot(X_here+points[s, 0], Y_here+points[s, 1], points[s, 2], color=color)

                newsimplices = np.array([s + simplices0 for s in hull.simplices])

                newpoints = np.array([ i+[X_here,Y_here,0] for i in points])

                return newpoints, newsimplices

            else:
                return np.empty(0), np.empty(0)


        matrix = self.intensity

        L = matrix.shape[0]
        W = matrix.shape[1]

        if plot:
            fig = plt.figure(figsize=(4.8, 4.8))
            ax = fig.add_subplot(111, projection="3d")

        "multiple sectors"
        NUM = 0
        ls = np.linspace(0,1,L_sectors+1)
        ws = np.linspace(0,1,L_sectors+1)
        k=0
        for i in range(L_sectors):
            for j in range(L_sectors):

                newpts, newsmpls = SectorHull(matrix, int(ls[i]*L),int(ls[i+1]*L),\
                                    int(ws[j]*W),int(ws[j+1]*W),NUM,rel_depth,'lime')
                NUM+=len(newpts)
                print(NUM)
                print(newpts.shape, newsmpls.shape)
                if newpts.shape[0]:
                    points2 = np.concatenate((points2,newpts)) if k!=0 else newpts
                    hullsimplices2 = np.concatenate((hullsimplices2,newsmpls)) if k!=0 else newsmpls
                    k+=1
                    
        print('points shape',points2.shape)
        print('simplices shape',hullsimplices2.shape)

        points2n=points2*5/np.max(points2)
        writeobj(points2n,hullsimplices2,out_file)

        self.objfile = out_file 

        if plot:
            ax.set_title("{} Convex Hull segments".format(L_sectors**2),color="#D3D3D3")
            ax.set_facecolor('#3e404e')

            set_axes_equal(ax)
            plt.axis('off')
            plt.show()


    # viewport_default = (0.8*np.array(pygame.display.set_mode().get_rect()[2:]))
    def MeshView(self,viewport=(2048, 1152)):
        '''MeshView: triangulated mesh view with OpenGL [ uses pygame ]

        Parameters
        ----------
        viewport : (int,int)
            viewport / screen (width, height) for display window (default: 80% your screen's width & height)
        '''
        viewer.objview(self.objfile, usemtl=False, viewport=viewport)