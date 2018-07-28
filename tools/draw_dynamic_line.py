# draw dynamic lines
import numpy as np
import random as random
from matplotlib import pyplot as plt
from matplotlib import animation
import scipy.misc
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 700), ylim=(-250, 100))
# line, = ax.plot([], [], lw=2)
plotlays, plotcols = [2,3,5], ["green", "red", "orange"]

n=5
a = []
b = []
c = []
for i in range(500):
    a.append(5)
    b.append(5)
    c.append(5)
with open('udacity_compare.txt', 'r') as f:
    for line in f.readlines():
        a.append(float(line.strip().split(',')[0])*10)
        b.append(float(line.strip().split(',')[1])*10)
        c.append(float(line.strip().split(',')[2])*10)

x = []
y = []
x1 = []
y1 = []
x2 = []
y2 = []
lines = []
img = scipy.misc.imread('/home/cardwing/Desktop/demo_file/1_background.png')

for index,lay in enumerate(plotlays):
    lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
    lines.append(lobj)

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([],[])
    return lines

# animation function.  This is called sequentially
def animate(i):
    print(i)
    if i <= 500:
        return tuple(lines)
    else:
        for num, line in enumerate(lines): 
            if num == 0:
                x.append(np.linspace(i,i+1,n))
                y.append(np.linspace(a[i],a[i+1],n))
                line.set_data(x,y)
            if num == 1:
                x1.append(np.linspace(i,i+1,n))
                y1.append(np.linspace(b[i],b[i+1],n))
                line.set_data(x1,y1)    
            if num == 2:
                x2.append(np.linspace(i,i+1,n))
                y2.append(np.linspace(c[i],c[i+1],n))
                line.set_data(x2,y2) 
        return tuple(lines)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, np.arange(0,len(a)-1) ,init_func=init,
                               interval=50, blit=True, repeat=False)
plt.imshow(img, zorder=0,  extent=[0.1, 700.0, -250.0, 100.0])
plt.show()
