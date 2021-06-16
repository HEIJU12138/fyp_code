import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

plt.figure("P-R Curve")
plt.title('ROC graph')
plt.xlabel('False Postive Rate')
plt.ylabel('True Postive Rate')
y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
y_scores = np.array([0.9, 0.35, 0.86, 0.47, 0.55, 0.56, 0.74, 0.62, 0.5, 0.86, 0.8, 0.47, 0.44, 0.67, 0.43, 0.4, 0.52, 0.4, 0.35, 0.1])
y_scores1=np.array([0.95,0.94,0.93,0.4,0.91,0.30,0.45,0.89,0.40,0.88,0.83,0.97,0.50,0.47,0.25,0.18,0.60,0.55,0.36,0.42])
fpr,tpr,threshold=roc_curve(y_true,y_scores)
fpr1,tpr1,threshold1=roc_curve(y_true,y_scores1)
plt.plot(fpr,tpr)
plt.plot(fpr1,tpr1)
plt.plot([0,1],[0,1],c='b',linestyle='--')
plt.text(0.6,0.4,"Random Classifier")
plt.text(0.2,0.5,"Curve 1")
plt.text(0.2,0.9,"Curve 2")
plt.show()