import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

x=[0,1,2,3]
y=[3,2,4,3]
plt.scatter(x,y,c='blue')
plt.plot([1,2],[2,4],c='black',linewidth=1.2)
plt.plot([0,1],[3,2],c='black',linewidth=1.2)

x1=[1,3]
y1=[2,3]
plt.plot(x1,y1,c='black',linewidth=1.2)

x2=[0.5,1.2,2]
y2=[2.5,2.4,2.5]
plt.scatter(x2,y2,c='red',marker='v')
plt.text(1,1.6,"$x_i$")
plt.text(0,2.6,"$x_{i1}$")
plt.text(0.4,2.2,"$s_1$")
plt.text(1.1,2.8,"$s_3$")
plt.text(2,2.1,"$s_2$")
plt.text(3,2.6,"$x_{i3}$")
plt.text(2,3.6,"$x_{i2}$")
plt.xlim([-1,4])
plt.ylim([0,6])
plt.show()