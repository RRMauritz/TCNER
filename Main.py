from BOW import nb, logrec, svm
from WordEmbedding import lstm
from AccuracyCV import accuracy_CV, acc_test
from DataPrep import X_train, X_test, y_train, y_test

# Performance------------------------------------------------------------:
k = 10

# Test the performances of the models based on k-fold CV on the train set
# print('CV accuracy NB =  ', accuracy_CV(X_train, y_train, k, bow_nb))
# print('CV accuracy LR =  ', accuracy_CV(X_train, y_train, k, logrec))
# print('accuracy SVM =  ', accuracy_CV(X_train, y_train, k, svm))
# print('accuracy LSTM =  ', accuracy_CV(X_train, y_train, k, lstm))

# Test performance of selected model + preprocessing on test set
print('Test accuracy NB =  ', acc_test(X_train, X_test, y_train, y_test, nb))
print('Test accuracy LR =  ', acc_test(X_train, X_test, y_train, y_test, logrec))
print('Test accuracy SVM =  ', acc_test(X_train, X_test, y_train, y_test, svm))
print('Test accuracy LSTM =  ', acc_test(X_train, X_test, y_train, y_test, lstm))
