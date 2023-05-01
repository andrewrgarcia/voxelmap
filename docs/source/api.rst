API Reference
================

Global Methods
-----------------
As the methods are several, below are only listed the most pertinent global methods of ``voxelmap``, 
in order of the lowest level to highest level of applications to 3-D modeling operations.and classified in
sub-sections:

Special 
..................
.. autoclass:: voxelmap.voxelwrite

.. autoclass:: voxelmap.objdraw

.. autoclass:: voxelmap.objarray
   

Load and Save 
.................
.. autoclass:: voxelmap.load_array

.. autoclass:: voxelmap.save_array

.. autoclass:: voxelmap.tojson

.. autoclass:: voxelmap.load_from_json

Array Manipulation
....................
.. autoclass:: voxelmap.resize_array

.. autoclass:: voxelmap.roughen

.. autoclass:: voxelmap.random_kernel_convolve


Mapping 
..................
.. autoclass:: voxelmap.MarchingMesh

.. autoclass:: voxelmap.MeshView



Local Methods to Model class
-------------------------------
.. autoclass:: voxelmap.Model
   :members: 
   
   .. automethod:: __init__


.. autosummary::
   :toctree: generated

   voxelmap
