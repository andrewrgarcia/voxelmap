'''main test module to run unit tests on library
running may be as simple as `python setup.py test` on main directory path, 
but check online documentation
'''
import voxelmap as vxm
import numpy as np

from matplotlib import cm


def test_pickle():
    '''test pickle save and load of made-up array'''
    arr = np.random.randint(0,10,(7,7,7))
    vxm.save_array(arr,'random-array')

    loaded_arr = vxm.load_array('random-array')

    print(loaded_arr)
    
def test_custom_voxel_colormap_save():
    '''test the custom voxel colormap (dictionary) generation and drawing
    model.hashblocksAdd() adds dictionary entries to custom voxel
     colormap to draw model with `voxels` coloring scheme
    '''

    arr = np.random.randint(0,10,(7,7,7))
    model = vxm.Model(arr)

    model.hashblocksAdd(1,'#84f348',0.8); model.hashblocksAdd(2,'#4874f3'); model.hashblocksAdd(3,'#32CD32') ;  model.hashblocksAdd(4,'#653c77',0.90)
    model.hashblocksAdd(5,'lime',0.75) ;  model.hashblocksAdd(6,'k',) ;  model.hashblocksAdd(7,'#e10af2',0.3)
    model.hashblocksAdd(8,'red',0.3); model.hashblocksAdd(9,'orange',0.2)

    model.draw('voxels')

    model.save('myModel.json')


def test_custom_voxel_colormap_load():

    model = vxm.Model()
    model.load('myModel.json')

    print(model.array)
    print(model.hashblocks)
    model.draw('voxels')


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
    model.draw('voxels')

    'defined rgb voxel colors'
    model2= vxm.Model()
    model2.XYZ =  model.XYZ
    model2.sparsity = 5

    model2.RGB = [ hex(np.random.randint(0.5e7,1.5e7))[2:] for i in range(10) ] 
    model2.load(coords=True)
    model2.array = np.transpose(model2.array,(2,1,0))

    model2.draw('voxels')



def test_goxeldog():
    'process dog.txt from Goxel'
    path = 'extra/dog.txt'

    model = vxm.Model()
    model.load(path)

    model.array = np.transpose(model.array,(2,1,0))

    model.draw('voxels')

    model.hashblocksAdd(1,'yellow',1)
    model.hashblocksAdd(2,'black',0.4)
    model.hashblocksAdd(3,'cyan',0.75)
    model.hashblocksAdd(4,'#000000')

    model.draw('voxels')

    model.colormap = cm.rainbow
    model.alphacm = 0.8

    model.draw('none')

def test_sphere():
    'sphere: stress graphics'

    '-- MAKE SPHERE MODEL --'
    path = 'extra/sphere.txt'
    sphere = vxm.Model()
    sphere.load(path)

    'coloring the white blocks of the pixelated sphere'
    for i in np.argwhere(sphere.array!=0):
        color =  np.random.randint(10)
        sphere.array[tuple(i)] = color

    'create hashmap of voxel colors'
    sphere.hashblocksAdd(1,'#84f348',0.8)
    sphere.hashblocksAdd(2,'#4874f3')
    sphere.hashblocksAdd(3,'#32CD32')
    sphere.hashblocksAdd(4,'#653c77',0.90)
    sphere.hashblocksAdd(5,'lime',0.75)
    sphere.hashblocksAdd(6,'k',)
    sphere.hashblocksAdd(7,'#e10af2',0.3)
    sphere.hashblocksAdd(8,'red',0.3)
    sphere.hashblocksAdd(9,'orange',0.2)

    savedhash = sphere.hashblocks        # save created hashmap of voxel colors (lines above this one)
    sphere.array = vxm.resize_array(sphere.array,(0.5,0.5,0.5))     #resize sphere (make smaller)
    sphere.draw('voxels')            

    '-- MAKE WEDGE MODEL --'
    mid = sphere.array.shape[2]//2
    wedge_array = sphere.array[mid:,mid:,:mid]          # slice above sphere into wedge 

    wedge = vxm.Model(wedge_array)
    wedge.hashblocks = savedhash       # used the hashmap from model 1
    wedge.draw('voxels')


def test_image():

    img = vxm.Image('extra/land.png')       # incorporate fake land topography .png file
    # img = vxm.Image('extra/donut_lores.png')       # incorporate fake land topography .png file

    img.make()                             # resized to 1.0x original size i.e. not resized (default)
    mapped_img = img.ImageMap(12)              # mapped to 3d with a depth of 12 voxels
    
    model = vxm.Model(mapped_img)
    
    model.array = np.transpose(np.flip(model.array),(2,0,1))

    # img.MarchingMesh(12)
    # img.MeshView()



def test_ImageMesh():

    img = vxm.Image('extra/land2.png')       # incorporate fake land topography .png file

    img.make()                             # resized to 1.0x original size i.e. not resized (default)

    img.ImageMesh('land.obj', 12, 3, 3, False, figsize=(10,10))
    # img.ImageMesh('land.obj', 12, 0.52, 1, True, verbose=True)

    # img.MeshView(wireframe=False, viewport=(1152, 1152))


def test_MarchingMesh():

    model = vxm.Model()
    model.load('extra/island.json')
    model.draw('none')

    array = model.array
    array = vxm.roughen(array,1)

    vxm.MarchingMesh(array,'isle.obj',True)
    vxm.MeshView('isle.obj')