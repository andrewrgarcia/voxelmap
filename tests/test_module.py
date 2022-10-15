'''main test module to run unit tests on library
running may be as simple as `python setup.py test` on main directory path, 
but check online documentation
'''
import voxelmap as vxm
import numpy as np
import matplotlib.pyplot as plt

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

    model.gradmap(cm.terrain,0.5)

    model.draw('nuclear')

def test_gradient_voxel_colormap2():
    '''test the linear gradient voxel colormap (dictionary) drawing'''

    arr = np.random.randint(0,10,(7,7,7))
    model = vxm.Model(arr)

    model.gradmap(cm.terrain,0.5)

    model.draw('linear')


def test_goxeldog():
    'process dog.txt from Goxel'
    path = 'extra/dog.txt'

    gox = vxm.Goxel(path)
    gox.update_colors('8f563b',1)
    gox.update_colors('ac3232',2)
    gox.update_colors('000000',3)
    gox.update_colors('ffffff',4)

    dog = gox.importfile()
    dog = np.transpose(dog,(2,1,0))

    model = vxm.Model(dog)

    model.customadd(1,'#8f563b',0.8)
    model.customadd(2,'#ac3232')
    model.customadd(3,'#000000')

    model.draw('voxels')


    model.gradmap(cm.terrain,0.5)
    model.draw('nuclear')

def test_sphere():
    'sphere: stress graphics'
    path = 'extra/sphere.txt'

    gox = vxm.Goxel(path)

    gox.update_colors('ffffff',1)   #update voxel colors (only ffffff -> white) from .txt to integer index in array

    sphere = gox.importfile()       #convert gox .txt to numpy array

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

    img = vxm.Image('extra/mountain.png')    # cat with glasses image (Credit: https://www.pictorem.com/profile/Tpencilartist)
    img.make(1)                       #resized to 0.3x original size (default)
    mapped_img = img.map3d(12)          # mapped to 3d with a depth of 10 voxels
    
    model = vxm.Model(mapped_img)
    model.array = np.transpose(np.flip(model.array),(2,0,1))

    model.gradmap(cm.terrain,0.5)
    model.draw('linear')


# test_pickle()
test_custom_voxel_colormap()
# test_gradient_voxel_colormap1()
# test_gradient_voxel_colormap2()
# test_goxeldog()
# test_sphere()
# test_image()