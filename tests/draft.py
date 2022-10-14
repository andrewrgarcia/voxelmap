'''
Used as a draft script. 
this is NOT a test module to run unit tests. 
Check test_module.py file for that. 
'''
import sys
sys.path.append('/home/andrew/voxelmap') 
import voxelmap as vxm
from matplotlib import cm


import numpy as np
import matplotlib.pyplot as plt

a = np.random.randint(0,10,(7,7,7))
vxm.save_array(a,'block_a')

b = vxm.load_array('block_a')

model = vxm.Model(b)
model.customadd(1,'#84f348',0.8)
model.customadd(2,'#4874f3')
model.customadd(3,'#32CD32')
model.customadd(4,'#653c77',0.90)
model.customadd(5,'lime',0.75)
model.customadd(6,'k',)
model.customadd(7,'#e10af2',0.3)
model.customadd(8,'red',0.3)
model.customadd(9,'orange',0.2)

# print(model.hashblocks)
model.gradmap(cm.terrain,0.5)

model.draw('voxels')
