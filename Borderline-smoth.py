import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

x=[0.4,0.3,0.35,0.45,0.25,0.3,0.4,0.5,0.5,0.4,0.36]
y=[1.3,0.9,0.8,1.4,1.15,1.1,1.0,1.25,1.05,1.15,1.16]
plt.scatter(x,y,c='blue',linewidths=5.0)
x1=[0.25,0.45,0.46,0.5,0.45,0.47]
y1=[1.00,1.25,1.2,1.21,0.95,1.03]
plt.scatter(x1,y1,c='red',marker='*',linewidths=5.0)

# for the C noise
plt.plot([0.25,0.25],[1.00,1.15],c='red',linewidth=1.0)
plt.plot([0.25,0.3],[1,.9],c='red',linewidth=1.0)
plt.plot([0.25,0.3],[1,.9],c='red',linewidth=1.0)
plt.plot([0.25,0.3],[1,1.1],c='red',linewidth=1.0)

#for A safe
plt.plot([0.4,0.45],[1.3,1.25],c='red',linewidth=1.0)
plt.plot([0.45,0.46],[1.25,1.2],c='red',linewidth=1.0)
plt.plot([0.45,0.5],[1.25,1.25],c='red',linewidth=1.0)
plt.plot([0.45,0.5],[1.25,1.21],c='red',linewidth=1.0)

# for B danger
plt.plot([0.47,0.45],[1.03,0.95],c='red',linewidth=1.0)
plt.plot([0.47,0.4],[1.03,1],c='red',linewidth=1.0)
plt.plot([0.47,0.5],[1.03,1.05],c='red',linewidth=1.0)


plt.xlim([0,0.8])
plt.ylim([0.5,1.5])
plt.text(0.2,1.0,'C')
plt.text(0.45,1.3,'A')
plt.text(0.47,1.07,'B')
plt.figure(figsize=(5, 5))
plt.show()