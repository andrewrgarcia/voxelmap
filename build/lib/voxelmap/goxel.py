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
        self.hashblocks = {}

    
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
        
        elems = goxeltxt.T

        'define voxel hashblocks dict from colors present in voxel file'
        model_colors = sorted(list(set(goxeltxt['rgb'])))
        for i in range(len(model_colors)):
            # self.hashblocks.update({i+1: model_colors[i] })
            self.hashblocks[i+1] = [ '#'+model_colors[i], 1]

        # print(self.hashblocks)

        'write array from .txt file voxel color values and locs'
        for i in range(len(goxeltxt)):
            x,y,z = elems[i][0:3].astype('int')
            rgb = '#'+elems[i][3]
            array[z,y,x] = [i for i in self.hashblocks if self.hashblocks[i][0]==rgb][0]
                    
        return array
