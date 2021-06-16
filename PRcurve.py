#coding:utf-8
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
plt.figure("P-R Curve")
plt.title('P-R graph')
plt.xlabel('Recall')
plt.ylabel('Precision')
#y_true为样本实际的类别，y_scores为样本为正例的概率
y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
y_scores = np.array([0.9, 0.35, 0.86, 0.47, 0.55, 0.56, 0.74, 0.62, 0.5, 0.86, 0.8, 0.47, 0.44, 0.67, 0.43, 0.4, 0.52, 0.4, 0.35, 0.1])
y_scores1=np.array([0.95,0.94,0.93,0.4,0.91,0.30,0.45,0.89,0.40,0.88,0.83,0.97,0.50,0.47,0.25,0.18,0.60,0.55,0.36,0.42])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision1, recall1 ,thresholds1= precision_recall_curve(y_true,y_scores1)
plt.plot(recall,precision)
plt.plot(recall1,precision1)
plt.text(0.4,0.8,"Curve A")
plt.text(0.7,0.9,"CUrve B")
plt.show()