Island Mesh
=================

.txt file source: https://raw.githubusercontent.com/andrewrgarcia/voxelmap/main/model_files/argisle.txt

.. code-block:: python

    #argisle.txt
    import voxelmap as vxm
    import numpy as np

    model= vxm.Model()
    model.load('argisle.txt')
    model.array = np.transpose(model.array,(2,1,0))     #rotate dog
    model.draw('custom',background_color='white')

    'to convert to mesh'
    model.array = vxm.resize_array(model.array,(5,5,5)) #make array larger before mesh transformation
    model.MarchingMesh()
    model.MeshView(color='pink',wireframe=True,background_color='white',alpha=1,viewport=[700,700])
