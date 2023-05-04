Voxel Model to 3-D Mesh
==========================

.txt file source: https://raw.githubusercontent.com/andrewrgarcia/voxelmap/main/model_files/skull.txt

.. code-block:: python

    #skull.py
    import voxelmap as vxm

    model = vxm.Model()

    model.load('extra/skull.txt')

    arr = model.array 


    model.array = model.array[::-1]

    'draw in standard voxel form'
    model.draw('voxels',wireframe=True, background_color='#3e404e',window_size=[700,700])

    'to convert to mesh'
    model.MarchingMesh()
    model.MeshView(wireframe=True,alpha=1,color=True,background_color='#b064fd',viewport=[700,700])

