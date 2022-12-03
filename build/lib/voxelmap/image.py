
import matplotlib.image as mpimg
import numpy as np
import cv2

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
