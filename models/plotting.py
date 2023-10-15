import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_history(history):
    plt.figure(figsize=(15, 5))

    # Subplot for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Model Accuracy')

    # Subplot for loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Model Loss')

    plt.tight_layout()
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

    colors = cycle(['blue', 'darkgreen', 'red', 'purple', 'gold',
                'gray', 'orange', 'cyan', 'pink', 'olive'])

    plt.figure(figsize=(15, 10))

    # Create a grid of 2x5 subplots to fit all 10 classes
    for i, color in zip(range(n_classes), colors):
        plt.subplot(2, 5, i+1)
        plt.plot(fpr[i], tpr[i], color=color, lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve of {class_names[i]} (area = {roc_auc[i]:.2f})')
    
    plt.tight_layout()
    plt.show()

def cm_plot(conf_matrix, labels):

    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, cmap='Blues', annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()