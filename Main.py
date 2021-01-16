from NaiveBayes import bow_nb
from LR import logrec
from SVM import svm
from WordEmbedding import lstm
from AccuracyCV import accuracy_CV
from DataPrep import X_train, X_test, y_train, y_test

# Performance------------------------------------------------------------:
k = 10

# Test the performances of the models based on k-fold CV on the train set
print('accuracy NB =  ', accuracy_CV(X_train, y_train, k, bow_nb))
print('accuracy LR =  ', accuracy_CV(X_train, y_train, k, logrec))
print('accuracy SVM =  ', accuracy_CV(X_train, y_train, k, svm))
print('accuracy LSTM =  ', accuracy_CV(X_train, y_train, k, lstm))
