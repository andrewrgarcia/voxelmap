'''Data import module'''
import numpy as np
import pandas

class Data:
    def __init__(self):
        '''Data structure.

        Parameters
        ----------
        -- FOR FILE PROCESSING --
        file : str
            file name and/or path for goxel txt file

        -- FOR XYZ COORDINATE ARRAY PROCESSING -- 
        xyz : np.array(float )
            an array containing the x,y,z coordinates of shape `number_voxel-locations` x 3 [for each x y z]
        rgb : list[str] 
            a list for the colors of every voxels in xyz array (length: `number_voxel-locations`)
        sparsity : float
            a factor to separate the relative distance between each voxel (default:10.0 [> 50.0 may have memory limitations])
        
        -- OTHER --
        hashblocks : dict[int]{str, float }
            a dictionary for which the keys are the integer values on the discrete arrays (above) an the values are the color (str) for the specific key and the alpha for the voxel object (float)
        '''
        self.file = 'placeholder.txt'

        self.xyz = np.zeros((10,3))
        self.rgb = []
        self.sparsity = 10.0

        self.hashblocks = {}

    
    def dfgox(self,file):
        '''Import Goxel file and convert to numpy array
        '''

        df = pandas.read_csv(file,\
        sep=' ',skiprows=(0,1,2), names=['x','y','z','rgb','none'])

        return df
    

    def dfXYZ(self,xyz,colorList,sparsity):
        '''Import Goxel file and convert to numpy array
        '''
        df = pandas.DataFrame(sparsity*xyz,columns=['x','y','z'])
        
        df['rgb'] = colorList if len(colorList) else len(df)*['a3ebb1']

        return df
            
    
    def importdata(self,type='file'):

        if type == 'file':
            df = Data().dfgox(self.file)
        else:
            df = Data().dfXYZ(self.xyz,self.rgb,self.sparsity)

        minx,miny,minz = df.min()[0:3]
        maxx,maxy,maxz = df.max()[0:3]
        
        df['z']+= (-minz)
        df['y']+= (-miny)
        df['x']+= (-minx)
        
        Z,Y,X = int(maxz-minz+1), int(maxy-miny+1), int(maxx-minx+1)
        
        array = np.zeros((Z,Y,X))
        
        elems = df.T

        'define voxel hashblocks dict from colors present in voxel file'
        model_colors = sorted(list(set(df['rgb'])))


        for i in range(len(model_colors)):
            # self.hashblocks.update({i+1: model_colors[i] })
            self.hashblocks[i+1] = [ '#'+model_colors[i], 1]

        # print(self.hashblocks)

        'write array from .txt file voxel color values and locs'
        for i in range(len(elems.T)):
            x,y,z = elems[i][0:3].astype('int')
            rgb = '#'+elems[i][3]
            array[z,y,x] = [i for i in self.hashblocks if self.hashblocks[i][0]==rgb][0]
                    
        return array
