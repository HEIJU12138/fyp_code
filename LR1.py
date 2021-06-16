from sklearn.datasets import make_classification
import matplotlib; matplotlib.use('TkAgg')
from imblearn.pipeline import make_pipeline,Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn import datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix,confusion_matrix, roc_auc_score,accuracy_score,precision_score, recall_score, f1_score

from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

X,y = datasets.make_classification (n_samples= 10 , n_features=2, n_informative= 2, n_redundant= 0, n_repeated= 0, n_classes= 2, n_clusters_per_class=1, class_sep=1,
flip_y=0,random_state=17)
print(X,y)
# print('Origin dataset shape:{}'.format(Counter(y)))

# # pre coding
# undersamp=RandomUnderSampler(random_state=2)
# oversamp=RandomOverSampler(random_state=2)
# smote=SMOTE()
# lr=LogisticRegression()
# # pipeline for each type of sampler
# pipeline_0=Pipeline([('lr',lr)])
# pipeline_1=Pipeline([('undersamp',undersamp),('lr',lr)])
# pipeline_2=Pipeline([('oversamp',oversamp),('lr',lr)])
# pipeline_3=Pipeline([('smote',smote),('lr',lr)])
#
# # split trian and test
# x_train, x_test, y_train, y_test =train_test_split(X,y, random_state=1,test_size=0.2,stratify=y)
#
# # loop to run all pipelines
# pipeline_list= [pipeline_0,pipeline_1,pipeline_2,pipeline_3]
#
# for num, pipeline in enumerate(pipeline_list):
#       print("Estimating Pipline {}".format(num))
#       pipeline.fit(x_train,y_train)
#       y_pred1=pipeline.predict(x_test)
#       y_pred2=pipeline.predict(x_train)
#       probs=pipeline.predict_proba(x_test)[:,1]
#       # print("Metrics for pipeline {}".format(num))
#       sns.set()
#       f, ax = plt.subplots()
#       print(f1_score(y_test, y_pred1), roc_auc_score(y_test, y_pred1))
# print(confusion_matrix(y_train, y_pred2))
#       print(accuracy_score(y_train, y_pred2),precision_score(y_train, y_pred2),recall_score(y_train, y_pred2),f1_score(y_train, y_pred2),roc_auc_score(y_train, y_pred2))
# print(confusion_matrix(y_test,y_pred1))

#       f1_score(y_test,y_pred1), roc_auc_score(y_test,y_pred1))
#       C2 = confusion_matrix(y_test, y_pred1, labels=[0, 1])
#       # # print(C2)
#       sns.heatmap(C2, annot=True, fmt='d',ax=ax)  # 画热力图
#       ax.set_title('confusion matrix for the test set')  # 标题
#       ax.set_xlabel('predicted label')  # x轴
#       ax.set_ylabel('true label')  # y轴
#
#       plt.show()