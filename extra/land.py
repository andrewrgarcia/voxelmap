#land.py
import voxelmap as vxm
import numpy as np

img = vxm.Image('./land.png')       # incorporate fake land topography .png file

img.make()                             # resized to 1.0x original size i.e. not resized (default)

img.ImageMesh('land.obj',  12, 0.52, 1, False, figsize=(10,10))

img.MeshView(viewport=(1152, 1152))
