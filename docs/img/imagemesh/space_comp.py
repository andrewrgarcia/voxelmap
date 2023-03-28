import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(100,1200**2)


j=[2,4,10,16,30]

# plt.style.use('dark_background')
# plt.figure(facecolor='#3e404e')
# plt.rcParams['axes.facecolor']='#3e404e'
# plt.rcParams['text.color']='#cccccc'
# plt.rcParams['xtick.color']='#cccccc'
# plt.rcParams['ytick.color']='#cccccc'
# mpl.rc('axes',edgecolor='#cccccc')

for i in j:
    plt.plot(x,x*i/3,'--',label="{} Convex Hull segments".format(i**2))

i=50
plt.semilogy(x,x*i/3,'--',label="{} Convex Hull segments".format(i**2),alpha=0.5)

plt.plot(x,x*12,label="voxel cloud $\mathcal{O}(12n)$",linewidth=2.5,color='k')
plt.plot(x,x,'.',label="point cloud $\mathcal{O}(n)$",linewidth=2.5,color='r')

plt.title("GPU space complexity - 10x10 to 1200x1200 image map methods")
plt.legend(title='segmented Convex Hulls methods $\mathcal{O}(n \sqrt{s}/3)$')

# plt.xlabel('number of pixels in image',color='#cccccc')
# plt.ylabel('number of faces in 3-D model',color='#cccccc')

plt.xlabel('number of pixels in image')
plt.ylabel('number of simplices in 3-D model')

plt.show()