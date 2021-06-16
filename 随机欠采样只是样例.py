import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


# 为了画个正态真麻烦
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
# plot normal  //no need from now on
np.random.seed(2)
Cn1 = np.random.normal(loc=0, scale=3, size=500)
Cn2 = np.random.normal(loc=2, scale=2, size=500)
x_0 = np.arange(min(Cn1), max(Cn1), 0.1)
x_1 = np.arange(min(Cn2), max(Cn2), 0.1)
y_0 = normfun(x_0, Cn1.mean(), Cn1.std())
y_1 = normfun(x_1, Cn2.mean(), Cn2.std())
# n0, = plt.plot(x_0, y_0, linestyle='--', c='b')
# n1, = plt.plot(x_1, y_1, linestyle='--', c='g')
nt0, = plt.plot(x_0, y_0 * 0.9, c='b')
nt1, = plt.plot(x_1, y_1 * 0.1, c='g')

# 随机生成数据,用来实验的, 进而进行random sampling 实验
np.random.seed(2)
orin0 = np.random.normal(loc=0, scale=3, size=90)
# we need to arrange the data
# 还得画个sampling后的分布图
reserved_rate=0.3
reduce_number=90*(1-reserved_rate) #调这个参数即可
index_0=np.random.choice(orin0.shape[0],int(reduce_number),replace=False)
reduced_data=orin0[index_0]
index_2=np.arange(orin0.shape[0])
index_2=np.delete(index_2,index_0)
kept_data=orin0[index_2]
kept_data1=np.arange(min(kept_data),max(kept_data),0.01)
kept_y=normfun(kept_data1,kept_data.mean(),kept_data.std())
p_priorfor0_new=(len(orin0)-reduce_number)/(len(orin0)-reduce_number+10)
p_priorfor1_new=1-p_priorfor0_new
kept0,=plt.plot(kept_data1,kept_y*p_priorfor0_new,linestyle=':')
kept1,=plt.plot(x_1,y_1*p_priorfor1_new,linestyle=':')
# the minority data
C1 = np.random.normal(loc=2, scale=2, size=10)
y0_kept= [-0.01] * len(kept_data)
y0_redu= [-0.01] * len(reduced_data)
y1 = [-0.01] * len(C1)

# take into account the proportions

# plot
s0 = plt.scatter(kept_data, y0_kept, s=10, c='b')
s01=plt.scatter(reduced_data,y0_redu,s=10)
s1 = plt.scatter(C1, y1, s=10, marker='D', c='g')
plt.ylim((-0.02, 0.25))
plt.xlim((-7.5,10))
# # plt.legend([s0, s01, s1,nt0, nt1,kept0,kept1],
#            ['examples reserved from $C_N$', 'examples removed from $C_N$','examples from $C_P$','True P(x|$C_N$)P($C_N$)',
#             'True P(x|$C_P$)P($C_P$)','estimated P(x|$C_N$)P($C_N$)','estimated P(x|$C_P$)P($C_P$)'], loc='upper left')
plt.title("Majority under-sampling ({} $*100\%$ reserved)".format(str(reserved_rate)))
plt.xlabel('x')
plt.show()