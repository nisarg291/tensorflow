import matplotlib.animation as animation
from matplotlib import style
import matplotlib.pyplot as plt
from collections import deque
import random
X = deque(maxlen=20)
X.append(1)
Y = deque(maxlen=20)
Y.append(1)
plt.style.use('fivethirtyeight')



fig1 = plt.figure()

ax1 = fig1.add_subplot(1,1,1)

def animate(p):
    plot_data = open('test.txt','r').read()

    line_data = plot_data.split('\n')
    x1=[]
    y1=[]
    while X[-1]+1<=20:
        X.append(X[-1]+1)
        Y.append(Y[-1]+Y[-1]*random.uniform(-0.1,0.1))
    for line in line_data:
        if len(line)>1:
            x,y = line.split(',')
            x1.append(x)
            y1.append(y)


        ax1.clear()
        # ax1.plot(X,Y)
        ax1.plot(x1,y1)

anime_data = animation.FuncAnimation(fig1, animate, interval = 1000)

plt.show()
