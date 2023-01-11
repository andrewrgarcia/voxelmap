# Usage in Poetry shell

If some incompatibility issues persist, the package may be run in a Poetry shell virtual environment. Poetry must first be installed [(install Poetry)](https://python-poetry.org/docs/#installation). After this, the repository must be cloned. Below are some instructions to do this:

```ruby
git clone https://github.com/andrewrgarcia/voxelmap.git
#OR git clone git@github.com:your-github=username/voxelmap.git
cd voxelmap
poetry shell 
```
After accessing the `poetry shell` within the voxelmap directory, directories may be changed. The below example has command line instructions to access the Desktop, create and access an Examples folder, and download Python files which use the `voxelmap` library. 

```ruby
cd ~/Desktop
mkdir voxelmapExamples; cd voxelmapExamples

URL="https://raw.githubusercontent.com/andrewrgarcia/voxelmap/main/extra/"  
   
for value in land.png land.py island.py dog.py; do wget $URL$value ; done

python dog.py
python island.py
python land.py
```
<!--**dog.py** 	voxel model with gradient coloring
**island.py** 	voxel model with integer-map coloring
**land.py**	ImageMesh 3-D mapping of fake land topography image-->

This approach creates a handy environment to use voxelmap locally. To update the repository, simply cd into voxelmap and run `git pull`. 
