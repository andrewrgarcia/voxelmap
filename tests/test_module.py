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

    model.customadd(1,'#84f348',0.8)
    model.customadd(2,'#4874f3')
    model.customadd(3,'#32CD32')
    model.customadd(4,'#653c77',0.90)
    model.customadd(5,'lime',0.75)
    model.customadd(6,'k',)
    model.customadd(7,'#e10af2',0.3)
    model.customadd(8,'red',0.3)
    model.customadd(9,'orange',0.2)

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

test_pickle()
test_custom_voxel_colormap()
test_gradient_voxel_colormap1()
test_gradient_voxel_colormap2()

