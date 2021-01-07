import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NaiveBayes import bow_nb
from LR import logrec
from SVM import svm
from WordEmbedding import lstm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from DataPrep import X_train, X_test, y_train, y_test

#y_pred_nb = bow_nb(X_train, X_test, y_train)
# y_pred_lr = logrec(X_train, X_test, y_train)
# y_pred_svm = svm(X_train, X_test, y_train)
y_pred_nn = lstm(X_train, X_test, y_train)

# Performance------------------------------------------------------------:


#print('accuracy NB =  %s' % accuracy_score(y_pred_nb, y_test))
# print('accuracy LR =  %s' % accuracy_score(y_pred_lr, y_test))
# print('accuracy SVM =  %s' % accuracy_score(y_pred_svm, y_test))
print('accuracy lstm = %s' % accuracy_score(y_pred_nn, y_test))

categories = ['INFOCOM', 'ISCAS', 'SIGGRAPH', 'VLDB', 'WWW']
y_pred = y_pred_nn
# print(classification_report(y_test, y_pred))
# # TODO: hoe komen we aan de micro-average precision ipv macro-average precision?
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=categories,
       yticklabels=categories, title="Confusion matrix")
plt.yticks(rotation=0)
plt.show()
