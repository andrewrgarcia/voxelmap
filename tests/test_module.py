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
    
def test_custom_voxel_colormap():
    '''test the custom voxel colormap (dictionary) generation and drawing
    model.customadd() adds dictionary entries to custom voxel
     colormap to draw model with `voxels` coloring scheme
    '''

    arr = np.random.randint(0,10,(7,7,7))
    model = vxm.Model(arr)

    model.customadd(1,'#84f348',0.8); model.customadd(2,'#4874f3'); model.customadd(3,'#32CD32') ;  model.customadd(4,'#653c77',0.90)
    model.customadd(5,'lime',0.75) ;  model.customadd(6,'k',) ;  model.customadd(7,'#e10af2',0.3)
    model.customadd(8,'red',0.3); model.customadd(9,'orange',0.2)

    model.draw('voxels')

def test_gradient_voxel_colormap1():
    '''test the nuclear gradient voxel colormap (dictionary) drawing'''

    arr = np.random.randint(0,10,(7,7,7))
    model = vxm.Model(arr)

    model.colormap = cm.terrain
    model.alphacm = 0.5

    model.draw('nuclear')

def test_gradient_voxel_colormap2():
    '''test the linear gradient voxel colormap (dictionary) drawing'''

    arr = np.random.randint(0,10,(7,7,7))
    model = vxm.Model(arr)

    model.colormap = cm.terrain
    model.alphacm = 0.5

    model.draw('linear')

def test_voxelcrds():

    data = vxm.Data()
    data.xyz = np.random.randint(-1,1,(10,3))+np.random.random((10,3))
    data.sparsity = 5
    
    'undefined rgb voxel colors'
    cubes = data.importdata('coords')
    cubes = np.transpose(cubes,(2,1,0))
    model = vxm.Model(cubes)
    model.hashblocks = data.hashblocks
    model.draw('voxels')

    'defined rgb voxel colors'
    data.rgb = [ hex(np.random.randint(0.5e7,1.5e7))[2:] for i in range(10) ] 
    cubes = data.importdata('coords')
    cubes = np.transpose(cubes,(2,1,0))
    model = vxm.Model(cubes)
    model.hashblocks = data.hashblocks
    model.draw('voxels')



def test_goxeldog():
    'process dog.txt from Goxel'
    path = 'extra/dog.txt'

    # gox = vxm.Goxel(path)
    data = vxm.Data()
    data.file = path

    dog = data.importdata()
    dog = np.transpose(dog,(2,1,0))

    model = vxm.Model(dog)

    model.hashblocks = data.hashblocks

    model.draw('voxels')

    model.customadd(1,'yellow',1)
    model.customadd(2,'black',0.4)
    model.customadd(3,'cyan',0.75)
    model.customadd(4,'#000000')

    model.draw('voxels')

    model.colormap = cm.rainbow
    model.alphacm = 0.8

    model.draw('nuclear')

def test_sphere():
    'sphere: stress graphics'
    path = 'extra/sphere.txt'

    data = vxm.Data()
    data.file = path

    sphere = data.importdata('file')       #convert gox .txt to numpy array

    'coloring the white blocks of the pixelated sphere'
    for i in np.argwhere(sphere!=0):
        color =  np.random.randint(10)
        sphere[tuple(i)] = color

    '--MAKE SPHERE MODE (model1)--'
    model1 = vxm.Model(sphere)

    'create hashmap of voxel colors'
    model1.customadd(1,'#84f348',0.8)
    model1.customadd(2,'#4874f3')
    model1.customadd(3,'#32CD32')
    model1.customadd(4,'#653c77',0.90)
    model1.customadd(5,'lime',0.75)
    model1.customadd(6,'k',)
    model1.customadd(7,'#e10af2',0.3)
    model1.customadd(8,'red',0.3)
    model1.customadd(9,'orange',0.2)
    savedhash = model1.hashblocks        # save created hashmap of voxel colors (lines above this one)

    # model1.draw('voxels')             # do not draw full sphere (keep tests relatively short)

    '--MAKE WEDGE MODEL (model2)--'
    mid = sphere.shape[2]//2
    wedge = sphere[mid:,mid:,:mid]          # slice above sphere into wedge 

    model2 = vxm.Model(wedge)
    model2.hashblocks = savedhash       # used the hashmap from model 1
    model2.draw('voxels')


def test_image():

    img = vxm.Image('extra/land.png')       # incorporate fake land topography .png file
    img.make(1)                             # resized to 1.0x original size i.e. not resized (default)
    mapped_img = img.map3d(12)              # mapped to 3d with a depth of 12 voxels
    
    model = vxm.Model(mapped_img)
    model.array = np.transpose(np.flip(model.array),(2,0,1))

    model.colormap = cm.terrain
    model.alphacm = 0.5

    model.draw('linear')


test_pickle()
test_custom_voxel_colormap()
test_gradient_voxel_colormap1()
test_gradient_voxel_colormap2()
test_goxeldog()
test_sphere()
test_image()
test_voxelcrds()
