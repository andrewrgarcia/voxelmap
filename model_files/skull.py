import voxelmap as vxm

model = vxm.Model()

model.load('./skull.txt')

arr = model.array 


model.array = model.array[::-1]

'draw in standard voxel form'
model.draw('custom',wireframe=True, background_color='#3e404e',window_size=[700,700])

'to convert to mesh'
model.MarchingMesh()
model.MeshView(wireframe=True,alpha=1,color=True,background_color='#b064fd',viewport=[700,700])
