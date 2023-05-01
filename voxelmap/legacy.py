# No longer used, but let's keep just in case

import numpy as np
from voxelmap.annex import *

def voxelwrite(array, filename = 'scene.obj'):
    '''
    Writes a 3-D voxel model from the provided, third-order (3-D) `array` as an .obj file
    
    Parameters
    ----------
    array : np.array(int)
            array of the third-order populated with discrete, non-zero integers which may represent a voxel block type
    filename : str
            name of .obj file to save model as. 
    '''


    # vertices diff (diffvs) for cube writing
    diffvs = np.array([
        [-0.50, -0.50, 0.00],
        [-0.50, 0.50, 0.00],
        [0.50, 0.50, 0.00],
        [0.50 ,-0.50, 0.00],
        [-0.50 ,-0.50 ,1.00],
        [ 0.50, -0.50, 1.00],
        [0.50, 0.50, 1.00],
        [-0.50 ,0.50 ,1.00]])
    


    with open(filename, 'w') as f:
        for coords in np.argwhere(array!=0):

            # z,y,x = coords
            for dverts in diffvs:
                # dz,dy,dx = verts
                
                posits = (dverts + coords)
            
                f.write("v  {:.4f} {:.4f} {:.4f}\n".format(*posits))


        block = """
vt 1.00 0.00 0.00 
vt 1.00 1.00 0.00
vt 0.00 1.00 0.00
vt 0.00 0.00 0.00

vn 0.00 0.00 -1.00
vn 0.00 0.00 1.00
vn 0.00 -1.00 0.00
vn 1.00 0.00 0.00
vn 0.00 1.00 0.00
vn -1.00 0.00 0.00

\n"""


       
        f.write("\n"+block)

        f.write("\ng Polyhedral\n\n")

        v_idcs_text = list('123 341 567 785 146 651 437 764 328 873 215 582')
        v_idcs = np.array([int(i) for i in v_idcs_text if i != ' '])

        for i in range(len(np.argwhere(array!=0))):

            f.write("""
f {}/1/1 {}/2/1 {}/3/1
f {}/3/1 {}/4/1 {}/1/1
f {}/4/2 {}/1/2 {}/2/2
f {}/2/2 {}/3/2 {}/4/2
f {}/4/3 {}/1/3 {}/2/3
f {}/2/3 {}/3/3 {}/4/3
f {}/4/4 {}/1/4 {}/2/4
f {}/2/4 {}/3/4 {}/4/4
f {}/4/5 {}/1/5 {}/2/5
f {}/2/5 {}/3/5 {}/4/5
f {}/4/6 {}/1/6 {}/2/6
f {}/2/6 {}/3/6 {}/4/6
""".format(*(v_idcs+i+(i*7))))
            
            # 0 -> 1
            # 1 -> 9 
            # 2 -> 17





def objdraw(array,filename='scene.obj',color='black',alpha=0.5,wireframe=False,wireframe_color='white',background_color='#cccccc', viewport = [1024, 768]):
    '''
    Creates a 3-D voxel model (.obj file) from the provided, third-order (3-D) `array`. It then uses the global method MeshView to draw the .obj file and display it on screen 
    
    Parameters
    ------------------
    array : np.array(int)
            array of the third-order populated with discrete, non-zero integers which may represent a voxel block type
    filename : str
            name of .obj file to save model as. 
    objfile: string
        .obj file to process with MeshView [in GLOBAL function only]
    wireframe: bool
        Represent mesh as wireframe instead of solid polyhedron if True (default: False). 
    color : string / hexadecimal
        mesh color. default: 'pink'
    alpha : float
        opacity transparency range: 0 - 1.0. Default: 0.5
    background_color : string / hexadecimal
        color of background. default: 'pink'
    viewport : (int,int)
        viewport / screen (width, height) for display window (default: 80% your screen's width & height)
    '''
    voxelwrite(array, filename = filename)
    # MeshView(objfile=filename,wireframe=wireframe,color=color,alpha=alpha,background_color=background_color, viewport = viewport)
    MeshView(filename,color,alpha,wireframe,wireframe_color,background_color, viewport )