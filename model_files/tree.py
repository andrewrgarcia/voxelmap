import numpy as np
import voxelmap as vxm

# Create a 20x20x20 array
tree_array = np.zeros((20, 20, 20), dtype=int)

# Define the trunk: a 2x2x10 column in the center
tree_array[5:15, 9:11, 9:11] = 1

# Define the leaves: a 10x10x10 cube on top of the trunk
tree_array[15:, 5:15, 5:15] = 2

# Initialize the model
model = vxm.Model(tree_array)

# Define colors for the trunk and leaves
model.hashblocks = {
    1: ['#8B4513', 1],  # Dark brown color for the trunk
    2: ['green', 1]     # Green color for the leaves
}

# Draw the model with PyVista VTK 
model.draw(coloring='custom', len_voxel=1, background_color='#ffffff')

# OR Draw it with matplotlib tools
model.draw_mpl(coloring='custom')

