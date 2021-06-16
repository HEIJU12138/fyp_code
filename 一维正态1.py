import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

#为了画个正态真麻烦
def normfun(x, mu, sigma):
  pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
  return pdf


# plot normal
np.random.seed(2)
Cn1=np.random.normal(loc=0,scale=2,size=500)
Cn2=np.random.normal(loc=2,scale=1,size=10000)
x_0=np.arange(min(Cn1),max(Cn1),0.1)
x_1=np.arange(min(Cn2),max(Cn2),0.1)
y_0=normfun(x_0,Cn1.mean(),Cn1.std())
y_1=normfun(x_1,Cn2.mean(),Cn2.std())
n0,=plt.plot(x_0,y_0,linestyle='--',c='b')
n1,=plt.plot(x_1,y_1,linestyle='--',c='g')

# 随机生成数据,用来实验的
np.random.seed(2)
C0=np.random.normal(loc=0,scale=2,size=90)
print(C0)
C1=np.random.normal(loc=2,scale=1,size=10)
y0=[-0.01]*90
y1=[-0.01]*10

#take into account the proportions
nc0,=plt.plot(x_0,y_0*0.9,c='b')
nc1,=plt.plot(x_1,y_1*0.1,c='g')

# predict(我觉得可以先用传统的贝叶斯来做，而且最好thesis里面有贝叶斯)


#plot
s0=plt.scatter(C0,y0,s=10,c='b')
s1=plt.scatter(C1,y1,s=10,marker='D',c='g')
plt.ylim((-0.05,0.5))
# plt.legend([n0,n1,s0,s1,nc0,nc1],['P(x|$C_N$)','P(x|$C_P$)','examples from $C_N$','examples from $C_P$','P(x|$C_N$)P($C_N$)','P(x|$C_P$)P($C_P$)'],loc='upper left')
plt.ylabel("Probability")
plt.xlabel('x')

plt.show()