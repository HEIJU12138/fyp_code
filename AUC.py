import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve,roc_auc_score,auc

plt.figure("P-R Curve")
plt.title('ROC graph')
plt.xlabel('False Postive Rate')
plt.ylabel('True Postive Rate')
y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
y_scores = np.array([0.95, 0.35, 0.86, 0.93, 0.55, 0.56, 0.74, 0.62, 0.5, 0.86, 0.8, 0.47, 0.44, 0.85, 0.43, 0.4, 0.52, 0.4, 0.35, 0.1])
y_scores1=np.array([0.95,0.56,0.93,0.4,0.51,0.30,0.45,0.89,0.40,0.83,0.83,0.97,0.70,0.47,0.25,0.18,0.32,0.55,0.36,0.42])
fpr,tpr,threshold=roc_curve(y_true,y_scores)
fpr1,tpr1,threshold1=roc_curve(y_true,y_scores1)
auc_score1=auc(fpr,tpr)
auc_score2=auc(fpr1,tpr1)
print(auc_score1,auc_score2)
plt.plot(fpr,tpr,c='b')
plt.plot(fpr1,tpr1)
plt.plot([0,1],[0,1],c='b',linestyle='--')
plt.text(0.5,0.4,"Random Classifier")
plt.text(0.4,0.7,"Curve 3")
plt.text(0.2,0.9,"Curve 4")
plt.show()