import numpy as np
import matplotlib.pyplot as plt
from timethis import timethis

import voxelmap as vxm


# result = vxm.load_array('terraformed')

# print(np.min(result),np.max(result))
# model = vxm.Model(file='docs/img/land.png')
# # model = vxm.Model(result)

# model.ImageMap(5)
# # model.ImageMesh(L_sectors=50,rel_depth = 0.5,plot=Falseze)
# model.MeshView(wireframe=False,color='lime',alpha=0.75)


# sub_matrices_rows = np.array_split(result, 3, axis=0)
# sub_matrices_cols = [np.array_split(sub_matrix, 3, axis=1) for sub_matrix in sub_matrices_rows]
# sectors = sub_matrices_cols

# for i in range(3):
#     for j in range(3):
#         plt.figure()
#         plt.axis('off')
#         plt.tight_layout()
#         plt.imshow(sectors[i][j],cmap='terrain')

# plt.show()


#import packages
import cv2
import matplotlib.pyplot as plt

plt.imshow(cv2.imread('docs/img/land.png'))      # display fake land topography .png file as plot
plt.axis('off')
plt.show()

#import packages
import numpy as np
from matplotlib import cm

model = vxm.Model(file='docs/img/land.png')             # incorporate fake land topography .png file to voxelmap.Image class

'''
# The image is then resized for the voxel draw with the matplotlib method i.e. ``Model().draw_mpl``. This is done with ``cv2.resize``, resizing the image from 1060x1060 to 50x50. After resizing, we convolve the image to obtain a less sharp color shift between the different gray regions with the ``cv2.blur`` method:
'''
print(model.array.shape)

model.array = cv2.resize(model.array, (50,50), interpolation = cv2.INTER_AREA)
print(model.array.shape)

model.array = cv2.blur(model.array,(10,10))    # blur the image for realiztic topography levels
plt.imshow(model.array)      # display fake land topography .png file as plot
plt.axis('off')
plt.show()


# model.ImageMap(12)              # mapped to 3d with a depth of 12 voxels

# model.MeshView()





model.ImageMesh('model.obj', 4 , 1, 1, False)

model.MeshView( alpha=0.7,background_color='#3e404e',color='white',viewport=(700, 700))