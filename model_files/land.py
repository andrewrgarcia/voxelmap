#land.py
import voxelmap as vxm
import numpy as np
import cv2 as cv

img = vxm.Model(file='../docs/img/land.png')           # incorporate fake land topography .png file
img.array = cv.blur(img.array,(100,100))    # blur the image for realiztic topography levels

# img.make()                                  # resized to 1.0x original size i.e. not resized (default)

array = img.ImageMap()

print(array)
# img.draw()

# img.ImageMesh('land.obj',  12, 14, 1, False, figsize=(10,10))

# img.MeshView( alpha=0.7,background_color='#3e404e',color='white',viewport=(700, 700))
