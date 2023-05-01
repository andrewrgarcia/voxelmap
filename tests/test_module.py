'''main test module to run unit tests on library
running may be as simple as `python setup.py test` on main directory path, 
but check online documentation
'''
import voxelmap as vxm
import numpy as np

import cv2
import matplotlib.pyplot as plt
from matplotlib import cm


def test_arrays_from_obj():

    array = vxm.objcast('model_files/sphere.obj',10)
    sphere_model = vxm.Model(array)
    sphere_model.draw(wireframe=True,voxel_spacing=(1,1,1))
    sphere_model.MarchingMesh()
    sphere_model.MeshView(wireframe=True,color='w',alpha=1)

    array = vxm.objcast('model_files/simple_cube.obj',0.5)
    cube_model = vxm.Model(array)
    cube_model.draw(wireframe=True,voxel_spacing=(1,1,1))

def test_pickle():
    '''test pickle save and load of made-up array'''
    arr = np.random.randint(0,10,(7,7,7))
    vxm.save_array(arr,'random-array')

    loaded_arr = vxm.load_array('random-array')

    print(loaded_arr)
    
def test_custom_voxel_colormap_save():
    '''test the custom voxel colormap (dictionary) generation and drawing
    model.hashblocks_add() adds dictionary entries to custom voxel
     colormap to draw model with `voxels` coloring scheme
    '''

    arr = np.random.randint(0,10,(7,7,7))
    model = vxm.Model(arr)

    model.hashblocks_add(1,'#84f348',0.8); model.hashblocks_add(2,'#4874f3'); model.hashblocks_add(3,'#32CD32') ;  model.hashblocks_add(4,'#653c77',0.90)
    model.hashblocks_add(5,'lime',0.75) ;  model.hashblocks_add(6,'k',) ;  model.hashblocks_add(7,'#e10af2',0.3)
    model.hashblocks_add(8,'red',0.3); model.hashblocks_add(9,'orange',0.2)

    model.draw('custom')

    model.save('myModel.json')


def test_custom_voxel_colormap_load():

    model = vxm.Model()
    model.load('myModel.json')

    print(model.array)
    print(model.hashblocks)
    model.draw('custom')


def test_gradient_voxel_colormap1():
    '''test the nuclear gradient voxel colormap (dictionary) drawing'''

    arr = np.random.randint(0,10,(7,7,7))
    model = vxm.Model(arr)

    model.colormap = cm.terrain
    model.alphacm = 0.5

    model.draw('cool')

def test_gradient_voxel_colormap2():
    '''test the linear gradient voxel colormap (dictionary) drawing'''

    arr = np.random.randint(0,10,(7,7,7))
    model = vxm.Model(arr)

    model.colormap = cm.terrain
    model.alphacm = 0.5

    model.draw('fire')

def test_voxelcrds():

    model= vxm.Model()
    model.XYZ = np.random.randint(-1,1,(10,3))+np.random.random((10,3))
    model.sparsity = 5
    
    'undefined rgb voxel colors'
    model.load(coords=True)

    model.array = np.transpose(model.array,(2,1,0))
    model.draw('custom')

    'defined rgb voxel colors'
    model2= vxm.Model()
    model2.XYZ =  model.XYZ
    model2.sparsity = 5

    model2.RGB = [ hex(np.random.randint(0.5e7,1.5e7))[2:] for i in range(10) ] 
    model2.load(coords=True)
    model2.array = np.transpose(model2.array,(2,1,0))

    model2.draw('custom')



def test_goxeldog():
    'process dog.txt from Goxel'
    path = 'model_files/dog.txt'

    model = vxm.Model()
    model.load(path)

    model.array = np.transpose(model.array,(2,1,0))

    model.draw('custom')

    model.hashblocks_add(1,'yellow',1)
    model.hashblocks_add(2,'black',0.4)
    model.hashblocks_add(3,'cyan',0.75)
    model.hashblocks_add(4,'#000000')

    model.draw('custom')

    model.colormap = cm.rainbow
    model.alphacm = 0.8

    model.draw('none')

def test_sphere():
    'sphere: stress graphics'

    '-- MAKE SPHERE MODEL --'
    path = 'model_files/sphere.txt'
    sphere = vxm.Model()
    sphere.load(path)

    'coloring the white blocks of the pixelated sphere'
    for i in np.argwhere(sphere.array!=0):
        color =  np.random.randint(10)
        sphere.array[tuple(i)] = color

    'create hashmap of voxel colors'
    sphere.hashblocks_add(1,'#84f348',0.8)
    sphere.hashblocks_add(2,'#4874f3')
    sphere.hashblocks_add(3,'#32CD32')
    sphere.hashblocks_add(4,'#653c77',0.90)
    sphere.hashblocks_add(5,'lime',0.75)
    sphere.hashblocks_add(6,'k',)
    sphere.hashblocks_add(7,'#e10af2',0.3)
    sphere.hashblocks_add(8,'red',0.3)
    sphere.hashblocks_add(9,'orange',0.2)

    savedhash = sphere.hashblocks        # save created hashmap of voxel colors (lines above this one)
    sphere.array = vxm.resize_array(sphere.array,(0.5,0.5,0.5))     #resize sphere (make smaller)
    sphere.draw('custom')            

    '-- MAKE WEDGE MODEL --'
    mid = sphere.array.shape[2]//2
    wedge_array = sphere.array[mid:,mid:,:mid]          # slice above sphere into wedge 

    wedge = vxm.Model(wedge_array)
    wedge.hashblocks = savedhash       # used the hashmap from model 1
    wedge.draw('custom')


def test_image():

    'display original land image'
    plt.imshow(cv2.imread('docs/img/land.png'))      # display fake land topography .png file as plot
    plt.axis('off')
    plt.show()

    model = vxm.Model(file='docs/img/land.png')             # incorporate fake land topography .png file to voxelmap.Image class

    'resize the image with cv2 tool'
    model.array = cv2.resize(model.array, (50,50), interpolation = cv2.INTER_AREA)
    # print(model.array.shape)

    'blur the image and display output'
    model.array = cv2.blur(model.array,(10,10))    # blur the image for realiztic topography levels
    plt.imshow(model.array)      # display fake land topography .png file as plot
    plt.axis('off')
    plt.show()
    # print(model.array.shape)

    'do ImageMap on treated image'
    # mapped_model = model.ImageMap(12)              # mapped to 3d with a depth of 12 voxels
    # model = vxm.Model(mapped_model)
    # model.array = np.flip(np.transpose(model.array))

    model.array  = model.ImageMap(12)              # mapped to 3d with a depth of 12 voxels

    model.colormap = cm.terrain
    model.alphacm = 0.5
    model.draw_mpl('linear',figsize=(15,12))

def test_ImageMesh0():
    model = vxm.Model(file='docs/img/land.png')   # incorporate fake land topography .png file

    print(model.array.shape)

    # model.ImageMesh(out_file='scene.obj', L_sectors = 20, trace_min=1, rel_depth = 20, figsize=(15,12), plot='mpl')
    model.ImageMesh(out_file='scene.obj', L_sectors = 20, trace_min=1, rel_depth = 20, figsize=(15,12), plot=False)

    model.array = cv2.blur(model.array,(50,50))    # blur the image for realiztic topography levels
    # model.ImageMesh(out_file='scene.obj', L_sectors = 20, trace_min=1, rel_depth = 20, figsize=(15,12), plot='mpl')
    model.ImageMesh(out_file='modelblurred.obj', L_sectors = 20, trace_min=1, rel_depth = 20, figsize=(15,12), plot=False)

    model.objfile = 'modelblurred.obj'
    model.MeshView()

def test_ImageMesh():

    model = vxm.Model(file='docs/img/galactic.png')       # incorporate fake land topography .png file


    # model.ImageMesh('land.obj', 12, 3, 3, False, figsize=(10,10))
    model.ImageMesh('land.obj', 12, 3, 1, True, verbose=True)

    # model.MeshView(wireframe=False, viewport=(1152, 1152))


def test_MarchingMesh():

    model = vxm.Model()
    model.load('model_files/island.json')
    model.draw('none')

    array = model.array
    array = vxm.roughen(array,1)

    vxm.MarchingMesh(array,'isle.obj',True)
    vxm.MeshView('isle.obj')


def test_extraMarch():

    model = vxm.Model()

    # model.load('model_files/skull.txt')
    model.load('model_files/chibi.txt')

    arr = model.array 


    # model.array = vxm.resize_array(model.array , mult=(2,2,2))
    # model.array = vxm.resize_array(model.array , mult=(0.5,0.5,0.5))

    # model.hashblocks = {-1: ['#ff0000', 1], 1: ['#0197fd', 1], 2: ['#816647', 1], 3: ['#98fc66', 1], 4: ['#eeeeee', 1], 5: ['#ffff99', 1]}


    model.array = model.array[::-1]
    # model.draw('custom',background_color='#c77575')
    model.draw('custom',wireframe=False, background_color='#3e404e',window_size=[700,700])

    model.MarchingMesh(step_size=1)
    model.MeshView(wireframe=True,alpha=1,color='black',background_color='#6a508b')

def build_checkerboard(w, h) :
      re = np.r_[ w*[0,1] ]              # even-numbered rows
      ro = np.r_[ w*[1,0] ]              # odd-numbered rows
      return np.row_stack(h*(re, ro))

def test_build_and_save2txt():

    A = build_checkerboard(16,16)

    A = A+1
    
    for i in range(200):
        A[tuple(np.random.randint(0,31,2))] = 3

    print(A.shape)
    A = A.reshape(1,32,32)

    model = vxm.Model(A)

    model.hashblocks={
                1: ['#000000',1],
                2: ['#ffffff',1],
                3: ['#BF40BF',1]

    }

    model.save('file.txt')

def test_loadTXT():

    model = vxm.Model()

    model.load('model_files/checkers.txt')

    model.MarchingMesh(plot=True)

    model.array = vxm.resize_array(model.array, (0.5,0.5,0.5))
    model.draw('custom',background_color='#00FF00')