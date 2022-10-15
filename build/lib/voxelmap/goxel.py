'''Goxel import module'''
import numpy as np
import pandas

class Goxel:
    def __init__(self,file):
        '''Goxel structure.

        Parameters
        ----------
        file : str
            file name and/or path for goxel file e.g. my_goxel_file.gox
        '''
        self.file = file
        self.colorkey = {
                        '33cc66': 1,
                        '339900': 2,
                        'ff99ff': 3,
                        'ffccff': 4,
                        }

    def update_colors(self,color,array_index):
        '''Update voxel colors (hashes) from Goxel file to be represented as `array_index` integers

        Parameters
        ----------
        color : str
            color in hexadecimal format of specific voxel color-block in Goxel
        array_index: int
            integer to represent specific voxel color-block from Goxel
        '''
        self.colorkey.update({color : array_index})
    
    def importfile(self):
        '''Import Goxel file and convert to numpy array
        '''
        goxeltxt = pandas.read_csv(self.file,\
        sep=' ',skiprows=(0,1,2), names=['x','y','z','rgb','none'])
    
        minx,miny,minz = goxeltxt.min()[0:3]
        maxx,maxy,maxz = goxeltxt.max()[0:3]
        
        goxeltxt['z']+= (-minz)
        goxeltxt['y']+= (-miny)
        goxeltxt['x']+= (-minx)
        
        Z,Y,X = int(maxz-minz+1), int(maxy-miny+1), int(maxx-minx+1)
        
        array = np.zeros((Z,Y,X))
        
        colorkey = self.colorkey

        print(colorkey)
        elems = goxeltxt.T
        
        for i in range(len(goxeltxt)):
            x,y,z = elems[i][0:3].astype('int')
            rgb = elems[i][3]
            array[z,y,x] = colorkey[rgb]
                    
        return array
