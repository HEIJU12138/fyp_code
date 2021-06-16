import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
########################################################################
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
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
proportion=0.9
size_C0=int(200*proportion)
size_C1=int(200*(1-proportion))
C0 = np.random.normal(loc=0, scale=2, size=size_C0)
y0 = [-1]*size_C0
C1 = np.random.normal(loc=2, scale=1, size=size_C1)
y1 = [1] *size_C1
C=np.append(C0,C1)
y=np.append(y0,y1)
C = StandardScaler().fit_transform(C.reshape(-1, 1))


# prediction
classifiers={
"LR":LogisticRegression(max_iter=10000),
"LDA":LinearDiscriminantAnalysis(),
"SVM":svm.LinearSVC(),
"C45": DecisionTreeClassifier() ,
"KNN":KNeighborsClassifier(3)
}  #只有cart，可以试一下，之后直接把c45代入即可

 # iterate over classifiers
i=1
acc=[]
auc=[]
f1=[]
for name, clf in classifiers.items():
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(C, y, random_state=i, test_size=0.3, stratify=y)
        oversamp = SMOTE(random_state=23)
        x_train, y_train = oversamp.fit_resample(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
        clf.fit(x_train.reshape(-1,1), y_train)
        y_pred = clf.predict(C.reshape(-1,1))
        a=accuracy_score(y.reshape(-1, 1), y_pred.reshape(-1, 1))
        acc.append(a)
        b=roc_auc_score(y.reshape(-1, 1), y_pred.reshape(-1, 1))
        auc.append(b)
        c=f1_score(y.reshape(-1, 1), y_pred.reshape(-1, 1))
        f1.append(c)
        i+=1
    print('the acc is %.5f for %s'%(np.mean(acc),name))
    print('the auc is %.5f for %s' % (np.mean(auc), name))
    print('the f1 is %.5f for %s' % (np.mean(f1), name))