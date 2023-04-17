import os
import numpy as np

def confusion_matrices(pred_label, true_label):
    CM = []
    for i in range(10):
        cm_i = {}
        is_i = (true_label == i)
        pred_i = (pred_label == i)

        cm_i['TP'] = (pred_i & is_i).sum()
        cm_i['TN'] = (~pred_i & ~is_i).sum()
        cm_i['FN'] = (is_i).sum() - cm_i['TP']
        cm_i['FP'] = (~is_i).sum() - cm_i['TN']

        CM.append(cm_i)
    return CM


def sensitivity(cm):
    return cm['TP'] / (cm['TP'] + cm['FN'])


def specificity(cm):
    return cm['TN'] / (cm['FP'] + cm['TN'])


def accuracy(pred_cluster, true_label, mapping):
    pred_label = mapping[pred_cluster]
    return (pred_label == true_label).sum() / len(true_label)


def print_metric(CM, converge_iter, error_rate, metric_path):
    f = open(os.path.join(metric_path, 'metric.txt'), 'w')
    for i in range(len(CM)):
        cm_i = CM[i]
        print(file=f)
        print(f"Confusion Matrix {i}:", file=f)
        print(f"                Predict number {i}  Predict not number {i}", file=f)
        print(f"Is number {i}           {cm_i['TP']}                  {cm_i['FN']}", file=f)
        print(f"Is not number {i}       {cm_i['FP']}                  {cm_i['TN']}", file=f)
        print(file=f)
        print(f"Sensitivity: (Successfully predict number {i}):     {sensitivity(cm_i)}", file=f)
        print(f"Specifity (Successfully predict not number {i}):    {specificity(cm_i)}", file=f)
        print(file=f)
        print("- " * 30, file=f)

    print(file=f)
    print(f"Total iterations to converge: {converge_iter}", file=f)
    print(f"Total error rate: {error_rate}", file=f)
    f.close()