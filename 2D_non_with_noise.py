import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
########################################################################
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE
########################################################################
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, auc, f1_score, accuracy_score
########################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from collections import Counter
from imblearn.pipeline import make_pipeline, Pipeline

##############################################################################################################################

# decide the proportion and the number
proportion = 0.9
number = 500
# for the negative class
np.random.seed(1)
mean = (1, 2)
cov = ([1, 0], [0, 1])

x = np.random.multivariate_normal(mean, cov, int(number * proportion))
y = [0] * int(number * proportion)

# for the positive case
np.random.seed(2)
mean1 = (3, 2)

x1 = np.random.multivariate_normal(mean1, cov, int(number - number * proportion))
size_noise=int(number*(1-proportion)*0.2)
size_x1=int(number - number * proportion)
y1 = [1] * (size_noise+size_x1)
# create the noise
x10_noise=np.random.normal(loc=0,scale=1,size=size_noise)
np.random.seed(3)
x11_noise=np.random.normal(loc=0,scale=1,size=size_noise)
a10=np.array(x1[0:size_noise,0])
b10=np.array(x10_noise)
x10=a10+b10
     # 同理做第二个
a11=np.array(x1[0:size_noise,1])
b11=np.array(x11_noise)
# 合并
x11=a11+b11
x2list=list(zip(x10,x11))
x2array=np.array(x2list)
x1=np.concatenate((x1,x2array))
# 生成数据
X = np.concatenate((x, x1))
X = StandardScaler().fit_transform(X)
y = np.concatenate((y, y1))
classifiers = {
    "LR": LogisticRegression(max_iter=100000),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": svm.LinearSVC(),
    "C45": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(3)
}  # 只有cart，可以试一下，之后直接把c45代入即可
# for loop to analyse the data
i = 1
j = 1
acc = []
auc = []
f1 = []
figure = plt.figure(figsize=(21, 4))
for name, clf in classifiers.items():
    for i in range(100):
        # remember that the spilt is before the sampling method
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
        # sampling methods,here we can change the sampling methods
        # sm = BorderlineSMOTE(random_state=2)
        # X_train, y_train = sm.fit_resample(X_train, y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X)
        a = accuracy_score(y, y_pred)
        acc.append(a)
        u = roc_auc_score(y, y_pred)
        auc.append(u)
        f = f1_score(y, y_pred)
        f1.append(f)
        i += 1
        if i!=23:
            continue
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .01 # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax = plt.subplot(1,5,j)
        ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
        ax.set_title(name)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        j+=1
    print('the acc is %.5f for %s' % (np.mean(acc), name))
    print('the auc is %.5f for %s' % (np.mean(auc), name))
    print('the f1 is %.5f for %s' % (np.mean(f1), name))
figure.subplots_adjust(left=.02, right=.98)
plt.show()