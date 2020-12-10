import matplotlib.pyplot as plt
from sklearn.metrics import auc


# Draw ROC curve:
def plot_roc_curve(fpr, tpr):
    plt.figure()
    lw = 2
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


# Draw PR curve:
def plot_pr_curve(recall, precision):
    plt.figure()
    lw = 2
    pr_auc = auc(precision, recall)
    plt.plot(precision, recall, color='darkorange', lw=lw,
             label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.title('P_R')
    plt.legend(loc="lower right")
    plt.show()
