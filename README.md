# voxelmap

A Python library for making voxel models from NumPy arrays.

<a href="https://andrewatcloud.com/voxelmap/"><img src="https://github.com/andrewrgarcia/voxelmap/blob/main/extra/wingeddog.png?raw=true" width="300"></a>

https://user-images.githubusercontent.com/10375211/205150749-98c0ec3b-cd05-4e16-afd8-de9cb0e88fc3.mp4

## Installation

```ruby
pip install voxelmap
```

## Contributing / Hacktoberfest

Meaningful contributions to the project are always welcome. This project is also active as a part of Hacktoberfest 2022. Before making a PR, please make sure to read the [CONTRIBUTING](./CONTRIBUTING.md) document.

You may use the Issues section of this repository if you'd like to propose some new ideas/enhancements or report a bug.

## Usage Examples

#### Map a 2-D image to a 3-D model

<img src="https://github.com/andrewrgarcia/voxelmap/blob/main/extra/land.png?raw=true" width="150" >

<img src="https://github.com/andrewrgarcia/voxelmap/blob/main/extra/land_mapped.png?raw=true" >

Code:

```ruby
import voxelmap as vxm
import numpy as np

img = vxm.Image('extra/land.png')       # incorporate fake land topography .png file
img.make(1)                             # resized to 1.0x original size i.e. not resized (default)
mapped_img = img.map3d(12)              # mapped to 3d with a depth of 12 voxels

model = vxm.Model(mapped_img)
model.array = np.transpose(np.flip(model.array),(2,0,1))

model.colormap = cm.terrain
model.alphacm = 0.5
model.draw('linear')
```

#### Process an exported .txt file from a [Goxel](https://goxel.xyz/) project

<a href="https://goxel.xyz/"><img src="https://github.com/andrewrgarcia/voxelmap/blob/main/extra/goxel.png?raw=true" width="350"></a><img src="https://github.com/andrewrgarcia/voxelmap/blob/main/extra/dog.png?raw=true" width="350"><img src="https://github.com/andrewrgarcia/voxelmap/blob/main/extra/dawg.png?raw=true" width="350"><img src="https://github.com/andrewrgarcia/voxelmap/blob/main/extra/rainbowd.png?raw=true" width="350">

Code:

```ruby
import voxelmap as vxm
import numpy as np

'''process dog.txt from Goxel'''
gox = vxm.Goxel('../extra/dog.txt')

dog = gox.importfile()      # turn txt file to array

dog = np.transpose(dog,(2,1,0))     #rotate dog

'load dog array to voxelmap Model'
model = vxm.Model(dog)

'color transfer from Goxel to Model'
model.hashblocks = gox.hashblocks
model.draw('voxels')

'draw with custom colors'
model.customadd(1,'yellow',1)
model.customadd(2,'black',0.4)
model.customadd(3,'cyan',0.75)
model.customadd(4,'#000000')

model.draw('voxels')

'draw with nuclear fill and terrain colormap'
model.colormap = cm.rainbow
model.alphacm = 0.8
model.draw('nuclear')
```

#### Draw a 3-D model from an array with custom voxel coloring scheme `voxels`

<img src="https://github.com/andrewrgarcia/voxelmap/blob/main/extra/randomarray.png?raw=true" width="200">

Code:

```ruby
import voxelmap as vxm
import numpy as np

#make a 4x4x4 integer array with random values between 0 and 9
array = np.random.randint(0,10,(7,7,7))

#incorporate array to Model structure
model = vxm.Model(array)

#add voxel colors and alpha-transparency for integer values 0 - 9 (needed for `voxels` coloring)

model.customadd(1,'#84f348',0.8); model.customadd(2,'#4874f3'); model.customadd(3,'#32CD32')  model.customadd(4,'#653c77',0.90); model.customadd(5,'lime',0.75) ;  model.customadd(6,'k',)  model.customadd(7,'#e10af2',0.3); model.customadd(8,'red',0.3); model.customadd(9,'orange',0.2)


#draw array as a voxel model with `voxels` coloring scheme
model.draw('voxels')
```

## Disclaimer: Use At Your Own Risk

This program is free software. It comes without any warranty, to the extent permitted by applicable law. You can redistribute it and/or modify it under the terms of the MIT LICENSE, as published by Andrew Garcia. See LICENSE below for more details.

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

**[MIT license](./LICENSE)** Copyright 2022 © <a href="https://github.com/andrewrgarcia" target="_blank">Andrew Garcia</a>.
