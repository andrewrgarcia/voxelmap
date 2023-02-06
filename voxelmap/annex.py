import json
import numpy as np
import pickle

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
      ax: a matplotlib axis, e.g., as output from plt.add_subplot().
    '''
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def arr2crds(array,mult):
    Z  = np.max(array)
    # return np.array([ [*i,Z-array[tuple(i)]] for i in np.argwhere(array)])
    return np.array([ [*i,-mult*array[tuple(i)]] for i in np.argwhere(array)])

def tensor2crds(tensor,mult):
    # return np.array([ [*i,Z-array[tuple(i)]] for i in np.argwhere(array)])
    return np.array([ [*i*mult] for i in np.argwhere(tensor)])


def load_array(filename):
    '''Loads a pickled numpy array with `filename` name'''
    return pickle.load( open(filename, "rb" ),encoding='latin1')

def save_array(array,filename):
    '''Saves an `array` array with `filename` name using the pickle module'''
    return pickle.dump(array,open(filename,'wb'))



def tojson(filename, array, hashblocks={}):

    dict = {}

    Z,Y,X = array.shape

    for i in ["hashblocks","size","coords","val"]:
        dict[i] = []    

    for k in range(Z):
        for j in range(Y):
            for i in range(X): 
                if array[k,j,i] != 0:
                    dict["coords"].append( [k,j,i] ) 
                    dict["val"].append( int(array[k,j,i]) )


    dict["size"] = [Z,Y,X]
    dict["hashblocks"] = hashblocks
    
    jsonobj = json.dumps(dict)
    # open file for writing, "w" 
    f = open(filename,"w")
    # write json object to file
    f.write(jsonobj)
    # close file
    f.close()


def load_from_json(filename):
    '''Load JSON file to object'''
    with open(filename) as f:
        data = f.read()
    return json.loads(data)