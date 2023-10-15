from sklearn.metrics import roc_curve, auc
import seaborn as sns
from itertools import cycle
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def roc_plot(y_test, y_prob, class_names):

    y_test = label_binarize(y_test, classes=range(10))

    n_classes = y_test.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'green', 'red', 'magenta', 'yellow',
                   'brown', 'orange', 'darkgreen', 'gold'])

    plt.figure(figsize=(8, 5))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC Curve of {class_names[i]} (area = {roc_auc[i]:.2f})')
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristics')
    plt.legend(loc="lower right")
    plt.show()


def cm_plot(conf_matrix, labels):

    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, cmap='Blues', annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()