#galactic.py
'''
This script makes the 3-D model file (.obj format) from the chosen image.
The .obj file made in the last line may then be imported to a graphic editing software such as Blender
or viewed with voxelmap i.e. adding img.MeshView() after the last line
'''

import voxelmap as vxm

img = vxm.Model(file='galactic.png')       # load image
img.objfile = 'galactic.obj'          # set name of 3-D model file (.obj) to be made
# img.MarchingMesh(10)                  # make 3-D model from image. 
img.ImageMap(12)   
img.MeshView()

