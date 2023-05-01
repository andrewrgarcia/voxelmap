import json
import numpy as np
import pickle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid
from scipy import ndimage 
import pyvista

import pandas

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

def matrix_toXY(array,mult):
    """
    Converts a 2D numpy array into an XYv coordinate matrix where v is the corresponding element in every x-y coordinate.

    Parameters
    ----------
    array : np.array(int,int)
        The 2D numpy array to be converted to an XYv coordinate matrix.
    mult : int or float
        The multiplication factor to be applied to the elements in the matrix.
    """

    Z  = np.max(array)
    return np.array([ [*i,mult*array[tuple(i)]] for i in np.argwhere(array)])

def tensor_toXYZ(tensor,mult):
    """
    Converts a 3D numpy array (tensor) into an XYZ coordinate matrix.

    Parameters
    ----------
    tensor : np.array(int,int,int)
        The 3D array (tensor) to be converted to XYZ coordinates.
    mult : int or float
        The multiplication factor to be applied to all the coordinates.
    """
    return np.array([ [*i*mult] for i in np.argwhere(tensor)])


def resize_array(array, mult=(2,2,2)):
    '''Resizes a three-dimensional array by the three dim factors specified by `mult` tuple. 
    Converts to sparse array of 0s and 1s   

    Parameters
    ----------
    array : np.array(int,int,int)
        array to resize
    mult: tuple(float,float,float)
        depth length width factors to resize array with. e.g 2,2,2 resizes array to double its size in all dims
    '''
    
    unique = np.unique(array)

    array = ndimage.zoom(array, mult).astype('int')
    crds_nonzero = np.argwhere(array > 0)
    src = array.copy()

    print(np.unique(src))
    array.fill(0)

    for k in crds_nonzero:
        if src[tuple(k)] in unique:
            array[tuple(k)] = src[tuple(k)]

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


def load_array(filename):
    '''Loads a pickled numpy array with `filename` name'''
    return pickle.load( open(filename, "rb" ),encoding='latin1')

def save_array(array,filename):
    '''Saves an `array` array with `filename` name using the pickle module'''
    return pickle.dump(array,open(filename,'wb'))



def tojson(filename, array, hashblocks={}):
    '''Save 3-D array and `hashblocks` color mapping as JSON file'''
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


def toTXT(filename='file.txt',array=[],hashblocks={}):

    Z,Y,X = array.shape

    header = ["# Goxel 0.10.4\n", "# One line per voxel\n", "# X Y Z RRGGBB\n"]
    with open(filename, "w") as file:
        # Writing data to a file
        file.writelines(header)
        for k in range(Z):
            for j in range(Y):
                for i in range(X): 
                    if array[k,j,i] != 0:
                        
                        hexcolor = hashblocks[array[k,j,i]][0]
                        ijk = list([-16,-16,0]+np.array([i,j,k]))
                        file.write("{} {} {} {} \n".format(*ijk, hexcolor[1:] ))

        



def load_from_json(filename):
    '''Load JSON file to object'''
    with open(filename) as f:
        data = f.read()
    return json.loads(data)


def writeobj_CH(points, hull_simplices, filename = 'this.obj'):
    '''Writes the triangulated image, which makes a 3-D mesh model, as an .obj file.
    *for Convex hull function/code'''

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


def writeobj_MC(filename = 'this.obj', *args):
    '''Writes the triangulated image, which makes a 3-D mesh model, as an .obj file.
    *for scikit-learn`s Marching Cubes (MC)'''
    verts, faces, normals, values = args

    with open(filename, 'w') as f:
        for i in verts:
            f.write("v  {:.4f} {:.4f} {:.4f}\n".format(*i))


        block = """
vt 1.00 0.00 0.00 
vt 1.00 1.00 0.00
vt 0.00 1.00 0.00
vt 0.00 0.00 0.00

"""
        for i in normals:
            f.write("vn  {:.4f} {:.4f} {:.4f}\n".format(*i))

        f.write("\n"+block)

        f.write("\ng Polyhedral\n\n")

        for j in faces:
            # the vertex texture (vt) triangle indices which color a specific simplex are [ currently ] being defined at random 
            rand_t0 = np.random.randint(4)


            j+=1    # hull simplices start at index 1 not 0 (this makes the correction)
            j1,j2,j3 = j

            facestr = [j1,(rand_t0+0)%4+1,j1,\
                    j2,(rand_t0+1)%4+1,j2,\
                    j3,(rand_t0+2)%4+1,j3 
                    ]

            f.write("f {}/{}/{} {}/{}/{} {}/{}/{}\n".format(*facestr))



def MarchingMesh(array, out_file='scene.obj', level=0, spacing=(1., 1., 1.), gradient_direction='descent', step_size=1, allow_degenerate=True, method='lewiner', mask=None, plot=False, figsize=(4.8,4.8) ):
    '''
    Marching cubes on sparse 3-D integer `voxelmap` arrays (GLOBAL)

    Parameters
    -------------
    array: np.array((int/float,int/float,int/float))
        3-D array for which to run the marching cubes algorithm   
    out_file : str
        name and/or path for Wavefront .obj file output. This is the common format for OpenGL 3-D model files (default: model.obj) 
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
    '''

    '''Adapted from: https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html'''
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(
                                    array, level=level, spacing=spacing, 
                                    gradient_direction=gradient_direction, step_size=step_size,
                                    allow_degenerate=allow_degenerate, method=method, mask=mask
                                    )

    'write wavefront .obj file for generated mesh'
    writeobj_MC(out_file, verts, faces, normals, values)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    if plot:
        MeshView(out_file)


def MeshView(objfile='scene.obj',color='black',alpha=1,wireframe=True,wireframe_color='white',background_color='#cccccc', viewport = [1024, 768]):
    '''
    Triangulated mesh view with PyVista (GLOBAL)
    
    Parameters
    --------------
    objfile: string
        .obj file to process with MeshView [in GLOBAL function only]
    color : string / hexadecimal
        mesh color. default: 'pink'
    alpha : float
        opacity transparency range: 0 - 1.0. Default: 0.5
    wireframe: bool
        Represent mesh as wireframe instead of solid polyhedron if True (default: False). 
    wireframe_color: string / hex 
        edges or wireframe colors
    background_color : string / hexadecimal
        color of background. default: 'pink'
    viewport : (int,int)
        viewport / screen (width, height) for display window (default: 80% your screen's width & height)
    '''
    # Define a custom theme for the 3D plot
    my_theme = pyvista.themes.DefaultTheme()
    my_theme.color = color
    my_theme.lighting = True
    my_theme.show_edges = True if wireframe else False
    my_theme.edge_color = wireframe_color
    my_theme.background = background_color


    mesh = pyvista.read(objfile)
    mesh.plot(theme=my_theme,opacity=alpha,window_size = viewport)
    



def xyz_to_sparse_array(df,hashblocks,spacing=1):
    '''
    Converts a pandas DataFrame df with columns 'x', 'y', 'z', and 'rgb' to a sparse 3D array. The function returns the array and a dictionary `hashblocks` that maps voxel colors to their corresponding index values in the array.

    Parameters
    -------------
    df : pandas DataFrame
        The input DataFrame containing 'x', 'y', 'z', and 'rgb' columns.
    hashblocks : dict
        A dictionary that maps voxel colors to their corresponding index values in the array.
    spacing: float
        Determines the distance between points in a point cloud. It can be adjusted to create a denser or sparser point cloud. If points are all less than 1, the sparse array will be unable to draw a model because sparse arrays have discrete dimensions, so changing them to a larger value may be necessary. Likewise, if separation is too large, this value can be set to a fractional number e.g. 0.5
    '''
    df[['x','y','z']] = spacing*df[['x','y','z']]

    minx, miny, minz = df.min()[0:3]
    maxx, maxy, maxz = df.max()[0:3]

    df['z'] += (-minz)
    df['y'] += (-miny)
    df['x'] += (-minx)

    Z, Y, X = int(maxz-minz+1), int(maxy-miny+1), int(maxx-minx+1)

    array =  np.zeros((Z, Y, X))

    elems = df.T

    'define voxel hashblocks dict from colors present in voxel file'
    model_colors = sorted(list(set(df['rgb'])))

    if isinstance(model_colors[0], str):
        for i in range(len(model_colors)):
            # hashblocks.update({i+1: model_colors[i] })
            hashblocks[i+1] = ['#'+model_colors[i], 1]
        print('Color list built from file!\nhashblocks =\n',hashblocks)

    'write array from .txt file voxel color values and locs'
    for i in range(len(elems.T)):
        x, y, z = elems[i][0:3].astype('int')

        if isinstance(model_colors[0], str):
            rgb = '#'+elems[i][3]
            array[z, y, x] = [
                i for i in hashblocks if hashblocks[i][0] == rgb][0]
        else:
            array[z, y, x] = elems[i][3]

    return array, hashblocks


def wavefront_to_xyz(filename='scene.obj'):
    '''
    Converts a Wavefront .obj file into a df pandas dataframe of vertex coordinates to be represented as a point cloud, where df has columns 'z', 'y', 'x', and 'rgb'. 

    Parameters
    ---------------
    filename : str, optional
        The path and name of the .obj file to be read. Default is 'scene.obj'.
    '''

    # Read the .obj file into a list of strings
    with open(filename) as f:
        lines = f.readlines()

    # Initialize an empty list to hold the vertex coordinates
    vertices = []

    # Iterate over each line in the .obj file
    for line in lines:
        # Check if the line starts with 'v' and contains 3 coordinates
        if line.startswith('v ') and len(line.split()) == 4:
            # Extract the x, y, and z coordinates as floats
            x, y, z = [float(coord) for coord in line.split()[1:]]
            # Append the coordinates to the vertices list
            vertices.append([z, y, x, "#ffffff"])  # placeholder for the rgb value

    # Convert the vertices list to a pandas dataframe with headers z, y, x, rgb
    df = pandas.DataFrame(vertices, columns=['z', 'y', 'x', 'rgb'])

    return df


def objcast(filename='scene.obj',spacing=1):
    '''
    Converts a Wavefront .obj file into a sparse, third-order (3-D) NumPy array to represent a point cloud model.

    Parameters
    ---------------
    filename : str 
        Path to the .obj file. Defaults to 'scene.obj'.
    spacing : float
        Distance between points in the point cloud. Can be adjusted to create a denser or sparser point cloud. Defaults to 1.
    '''

    df = wavefront_to_xyz(filename)
    hashblocks={}
    array, _ = xyz_to_sparse_array(df,hashblocks,spacing)

    return array