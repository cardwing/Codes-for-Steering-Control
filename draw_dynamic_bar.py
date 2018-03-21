# draw dynamic lines
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.misc

fig = plt.figure()

position = np.arange(2)*300 + 400 

plt.tick_params(axis = 'x', colors = '#072b57')
plt.tick_params(axis = 'y', colors = '#072b57')
img = scipy.misc.imread('/home/cardwing/Desktop/demo_file/1_background.png')
speed = []

with open('/home/cardwing/Desktop/demo_file/demo_udacity_error_2.txt', 'r') as f:
    for line in f.readlines():
        speed.append([float(line.strip().split(',')[0]), float(line.strip().split(',')[1])])


heights = [0, 0]
rects = plt.bar(position, heights, align = 'center', color = 'white', width = 100) 
plt.xticks(position, ('A', 'B'))

plt.xlabel('X Axis', color = '#072b57')
plt.ylabel('Y Axis', color = '#072b57')
plt.title('My Chart', color = '#072b57')

plt.ylim((0,1100))
plt.xlim((0,1100))

# plt.grid(True)

rs = [r for r in rects]

def init():
    return rs

def animate(i):
    global rs, heights
    '''if all(map(lambda x: x==100, heights)):
        heights = [0, 0]
    else:'''
    print(i)
    if i >= 701 or i <= 500:
        heights = heights
    else:
        heights = [min(h+s,1100) for h,s in zip(heights,speed[i-500])]
    for h,r in zip(heights,rs):
        r.set_height(h)
    return rs

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=10000, interval=50, blit=True)
plt.imshow(img, zorder=0,  extent=[0.1, 1100.0, 0.0, 1100.0])
plt.show()
