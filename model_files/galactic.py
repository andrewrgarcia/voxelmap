#galactic.py
'''
This script makes the 3-D model file (.obj format) from the chosen image.
The .obj file made in the last line may then be imported to a graphic editing software such as Blender
or viewed with voxelmap i.e. adding img.MeshView() after the last line
'''

import voxelmap as vxm

img = vxm.Image('galactic.png')       # load image 
img.objfile = 'galactic.obj'          # set name of 3-D model file (.obj) to be made
img.resize(0.50)                      # resize image to 25% its original size for feasible file size rendering
img.MarchingMesh(50)                  # make 3-D model from image. 
# img.MeshView()