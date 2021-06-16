import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
########################################################################
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE
########################################################################
from sklearn.metrics import roc_auc_score,roc_curve,plot_roc_curve,auc,f1_score,accuracy_score
########################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from collections import Counter
from imblearn.pipeline import make_pipeline,Pipeline

##############################################################################################################################
# 随机生成数据,用来实验的, 进而进行random sampling 实验
np.random.seed(2)
C0 = np.random.normal(loc=0, scale=2, size=180)
y0 = [-1]*180
C1 = np.random.normal(loc=12, scale=1, size=20)
y1 = [1] *20

#random sampling is to make them balance
C=np.append(C0,C1)
y=np.append(y0,y1)
# 先划分数据集,这里可以改sampling method
x_train, x_test, y_train, y_test =train_test_split(C,y, random_state=23,test_size=0.3,stratify=y)
oversamp=BorderlineSMOTE(random_state=23)
x_train,y_train=oversamp.fit_resample(x_train.reshape(-1,1),y_train.reshape(-1,1))

# prediction
classifiers={
"LR":LogisticRegression(max_iter=10000),
"SVM":svm.LinearSVC(),
"KNN":KNeighborsClassifier(3),
"LDA":LinearDiscriminantAnalysis(),
"Cart": DecisionTreeClassifier() }  #只有cart，可以试一下，之后直接把c45代入即可

class1={}
 # iterate over classifiers
for name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    y_pred=clf.predict(C.reshape(-1,1))
    acc=accuracy_score(y.reshape(-1,1),y_pred.reshape(-1,1))
    auc=roc_auc_score(y.reshape(-1,1),y_pred.reshape(-1,1))
    f1=f1_score(y.reshape(-1,1),y_pred.reshape(-1,1))
    print("Accuracy for %s is %.1f%%"%(name,acc*100))
    print("AUC for %s is %.1f"%(name,auc))
    # score = clf.score(x_test.reshape(-1,1), y_test.reshape(-1,1))
    # class1[name]=plot_roc_curve(clf,x_test.reshape(-1,1),y_test.reshape(-1,1),name='ROC for {}'.format(name))

# plt.show()
