
import matplotlib.image as mpimg
import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
# from scipy.spatial import Delaunay

from voxelmap.annex import *


def findcrossover(array,low,high,value):
    'finds crossover index of array for `value` value'
    if array[high] <= value:
        return high
    if array[low] > value:
        return low 

    middle = (high+low) // 2        # floor-division (indexes must be integers)

    if array[middle] == value:
        return middle
    elif array[middle] < value:
        findcrossover(array,middle+1,high,value)
    
    return findcrossover(array,low,middle-1,value)


def findclosest(array, value): 
    'adapted from: https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/'      
    idx = (np.abs(array - value)).argmin()
    return idx

def mat2crds(matrix):
    # X,Y = matrix.shape
    # Z = np.max(matrix,type=int)
    crds  = [ [*i,matrix[tuple(i)]] for i in np.argwhere(matrix)]
    return crds

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

    def ImageMesh(self, out_file='object.obj', plot = True, L_sectors = 4, rel_depth = 0.52, trace_min = 5 ):

        print('''\nIMAGEMESH: 3-D triangulation of 2-D images with a Convex Hull algorithm
Andrew Garcia, 2022\n''')

        def writeobj(points, hull_simplices, filename = 'this.obj'):
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
                    rand_t0 = np.random.randint(4)
                    rand_n = np.random.randint(1,7)
                    # rand_n = 2

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
            '''SectorHull adapted [ significantly ] from 
            https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud'''

        
            sector = array[X_here:X_there,Y_here:Y_there]

            if len(np.unique(sector)) > trace_min:
            # if np.sum(sector)>0:
                points = arr2crds(sector,rel_depth)

                hull = ConvexHull(points)
                # print(hull.simplices)


                # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
                if plot:
                    for s in hull.simplices:
                        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                        ax.plot(X_here+points[s, 0], Y_here+points[s, 1], points[s, 2], color=color)

                newsimplices = np.array([s + simplices0 for s in hull.simplices])
                # newsimplices = simplices0 + hull.simplices

                newpoints = np.array([ i+[X_here,Y_here,0] for i in points])
                # newpoints = points

                return newpoints, newsimplices

            else:
                return np.empty(0), np.empty(0)






        matrix = self.intensity

        # matrix*=Z_stretch

        # points = arr2crds(matrix)

        # hull = Delaunay(pts)



        # Plot defining corner points
        # ax.scatter(points.T[0], points.T[1], points.T[2], color="lime", marker ="o", s=1
        # # , markeredgecolor='black'
        # )


        # pointsn=points*5/np.max(points)
        # # pointsn = points
        # hull = ConvexHull(points)
        # print('points shape',pointsn.shape)
        # print('simplices shape',hull.simplices.shape)




        L = matrix.shape[0]
        W = matrix.shape[1]

        # ax.set_title("sectors : 0")
        # newpoints, newsimplices = SectorHull(matrix, 0,L, 0,W)

        # newpointsn=newpoints*5/np.max(newpoints)
        # writeobj(newpointsn, newsimplices,"this0.obj")

        if plot:
            fig = plt.figure(figsize=(4.8, 4.8))
            ax = fig.add_subplot(111, projection="3d")

        "multiple sectors"
        # L_sectors = 8
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


        if plot:

            ax.set_title("{} Convex Hull segments".format(L_sectors**2),color="#D3D3D3")
            # ax.set_facecolor((0.25,0.25,0.25))
            ax.set_facecolor('#3e404e')


            # Make axis label
            # for i in ["x", "y", "z"]:
            #     eval("ax.set_{:s}label('{:s}')".format(i, i))

            # print(hull.simplices)
            # print('len(hull.simplices) ',hull.simplices.shape)
            # print('len(pts) ',len(pts))
            set_axes_equal(ax)
            plt.axis('off')
            plt.show()

            'show image'
            # plt.figure(figsize=(4.8, 4.8))
            # plt.imshow(matrix,cmap='bone',)
            # plt.show()

