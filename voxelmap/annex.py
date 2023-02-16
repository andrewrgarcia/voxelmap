import json
import numpy as np
import pickle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid


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


def MarchingMesh(
                array, out_file='model.obj', level=0,
                spacing=(1., 1., 1.), gradient_direction='descent', step_size=1, 
                allow_degenerate=True, method='lewiner', mask=None,
                plot=False, figsize=(4.8,4.8) 
                ):
    '''[GLOBAL_METHOD] 
    Marching cubes on sparse 3-D integer `voxelmap` arrays

    Parameters
    ----------
    array: np.array((int/float,int/float,int/float))
        3-D array for which to run the marching cubes algorithm   
    out_file : str
        name and/or path for Wavefront .obj file output. This is the common format for OpenGL 3-D model files (default: model.obj) 

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
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        print(verts[faces-1])
        mesh = Poly3DCollection(verts[faces-1])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        def maxmin(arr): return np.min(arr), np.max(arr)

        ax.set_xlim(*maxmin(verts.T[0]))  
        ax.set_ylim(*maxmin(verts.T[1])) 
        ax.set_zlim(*maxmin(verts.T[2])) 

        plt.tight_layout()
        plt.show()



import voxelmap.objviewer as viewer

def MeshView(objfile='model.obj', wireframe=False, viewport=(2048, 1152)):
    '''[GLOBAL_METHOD] 
    MeshView: triangulated mesh view with OpenGL [ uses pygame ]

    Parameters
    ----------
    objfile: string
        .obj file to process with MeshView [in GLOBAL function only]
    wireframe: bool
        Represent mesh as wireframe instead of solid polyhedron if True (default: False). 
    viewport : (int,int)
        viewport / screen (width, height) for display window (default: 80% your screen's width & height)
    '''
    viewer.objview(objfile, wireframe=wireframe, usemtl=False, viewport=viewport)