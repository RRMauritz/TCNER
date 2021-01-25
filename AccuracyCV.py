import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support


def accuracy_CV(X, y, k, classifier):
    """"
    Determine the accuracy of the classifier based on k-fold CV on (X,y)
    :returns mean of precision, recall and f1-score over the k folds.
    Note, these are all captured by one number as they're the same in a multi-class problem
    """

    metrics = np.zeros((k, 3))
    kf = KFold(n_splits=k)
    i = 0
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        y_pred = classifier(X_train, X_val, y_train)
        # Note: we only take the first element as prec, recall and f1 are the same in the multi-class setting!
        metrics[i] = precision_recall_fscore_support(y_val.to_numpy(), y_pred, average='micro')[0]
        i = i + 1
    return np.mean(metrics)


def acc_test(X_train, X_test, y_train, y_test, classifier):
    # Train the classifier on (X_train,y_train) and then feed X_test to predict
    y_pred = classifier(X_train, X_test, y_train)
    # Evaluate on X_pred
    return precision_recall_fscore_support(y_test.to_numpy(), y_pred, average='micro')[0]
