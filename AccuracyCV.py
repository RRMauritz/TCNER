import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from NaiveBayes import bow_nb


def accuracy_CV(X, y, k, classifier):
    accuracies = np.zeros(k)
    kf = KFold(n_splits=k)
    i = 0
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        y_pred = classifier(X_train, X_val, y_train)
        accuracies[i] = accuracy_score(y_pred, y_val)
        i = i + 1
    return sum(accuracies) / len(accuracies)
