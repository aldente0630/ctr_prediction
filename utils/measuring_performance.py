import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
plt.style.use('seaborn')


def get_norm_entropy(y_true, y_score, eps=1e-08):
    p = y_true.mean()
    y_score = np.where(y_score < eps, eps, y_score)
    y_score = np.where(y_score > 1.0 - eps, 1.0 - eps, y_score)
    return log_loss(y_true, y_score) / (-1.0 * (p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def get_threshold_at_precision(y_true, y_score, precision):
    sorted_index = (-1.0 * y_score).argsort()
    precisions = y_true[sorted_index].cumsum() / (np.arange(y_true.shape[0]) + 1)
    return y_score[sorted_index][(precisions > precision).sum()]


def get_y_pred(y_score, threshold=0.5):
    return np.where(y_score >= threshold, 1, 0)


def plot_class_density(y_true, y_score, threshold=0.5, class_names=('0', '1')):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(y_score[y_true.astype('int') == 1], shade=True, linewidth=0.8, label=class_names[1], ax=ax)
    sns.kdeplot(y_score[y_true.astype('int') == 0], shade=True, linewidth=0.8, label=class_names[0], ax=ax)
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Score')
    ax.legend(title='Class', loc='best')
    
    
def plot_calibration_curve(y_true, y_score):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_score, n_bins=20)
    ax.plot(mean_predicted_value, fraction_of_positives, color='mediumblue', marker='s', label='Model', linewidth=1.0)
    ax.plot([0.0, 1.0], [0.0, 1.0], color='orange', linestyle='--', label='Perfectly Calibrated', linewidth=0.8)
    ax.legend(loc='best')
    ax.set_title('Calibration Plot (Reliability Curve)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Predicted Value')
    ax.set_ylabel('Fraction of Positives')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    
def plot_confusion_matrix(y_true, y_pred, normalize=False, class_names=('0', '1')):
    conf_mat = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    fmt = 'd'
    if normalize:        
        conf_mat = conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    sns.heatmap(conf_mat, cmap='coolwarm', annot=True, fmt=fmt, linewidths=0.5, square=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title('Confusion Matrix')
    return conf_mat


def plot_lift_curve(y_true, y_score):
    tested_sample_percent = (np.arange(len(y_true)) + 1) / len(y_true)
    found_sample_percent = y_true[(-1.0 * y_score).argsort()].cumsum() / y_true.sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tested_sample_percent, found_sample_percent, color='mediumblue', label='Lift Curve', linewidth=1.0)
    ax.fill_between([0.0, y_true.sum() / (len(y_true)), 1.0], [0.0, 1.0, 1.0], [0.0, y_true.sum() / (len(y_true)), 1.0], 
                    alpha=0.3, color='lightsteelblue')
    ax.set_xlabel('% Samples Tested')
    ax.set_ylabel('% Samples Found')
    ax.legend(loc='best')
    ax.set_title('Lift Chart', fontsize=12, fontweight='bold')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])


def plot_pr_curve(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color='mediumblue', linewidth=1.0, label='PR Curve (AUPRC: {0:0.4%})'.format(auprc))
    ax.fill_between(recall, precision, step='mid', alpha=0.3, color='lightsteelblue')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='best')
    ax.set_title('Precision - Recall', fontsize=12, fontweight='bold')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0, 1.01])
    return auprc


def plot_roc_curve(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='mediumblue', linewidth=1.0, label='ROC Curve (AUROC: {0:0.4%})'.format(auroc))
    ax.plot([0.0, 1.0], [0.0, 1.0], color='orange', linestyle='--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='best')
    ax.set_title('Receiver Operating Characteristic', fontsize=12, fontweight='bold')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0, 1.01])
    return auroc
