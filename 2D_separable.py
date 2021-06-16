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
mean1 = (8, 2)

x1 = np.random.multivariate_normal(mean1, cov, int(number - number * proportion))
y1 = [1] * int(number - number * proportion)

# 生成数据
X = np.concatenate((x, x1))
y = np.concatenate((y, y1))
X = StandardScaler().fit_transform(X)

classifiers = {
    "LR": LogisticRegression(max_iter=100000),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": svm.LinearSVC(),
    "C45": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(3)
}
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
        # # sampling methods,here we can change the sampling methods
        sm = BorderlineSMOTE(random_state=2)
        X_train, y_train = sm.fit_resample(X_train, y_train)
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
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# h = .01 # step size in the mesh
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(1, figsize=(5, 4))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.show()