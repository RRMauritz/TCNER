import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NaiveBayes import bow_nb
from LR import logrec
from SVM import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Load training data --------------------------------
with open('Data\DBLPTrainset.txt') as f:
    lines = f.readlines()

for i in range(len(lines)):
    pre = lines[i].split()[1:]
    lines[i] = [pre[0], ' '.join(pre[1:])]

# Store in pandas data frame
train = pd.DataFrame(lines, columns=['Label', 'Title'])

# Load test data------------------------------------------------
with open('Data\DBLPTestset.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    pre = lines[i].split()[1:]
    lines[i] = [' '.join(pre[1:])]

test = pd.DataFrame(lines, columns=['Title'])

with open('Data\DBLPTestGroundTruth.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()[1:][0]
test['Label'] = lines

X_train = train.Title
X_test = test.Title
y_train = train.Label
y_test = test.Label


# TODO: add pre-processing functions

y_pred_nb = bow_nb(X_train, X_test, y_train)
y_pred_lr = logrec(X_train, X_test, y_train)
y_pred_svm = svm(X_train, X_test, y_train)



# Performance------------------------------------------------------------:

categories = y_train.unique()

print('accuracy NB =  %s' % accuracy_score(y_pred_nb, y_test))
print('accuracy LR =  %s' % accuracy_score(y_pred_lr, y_test))
print('accuracy SVM =  %s' % accuracy_score(y_pred_svm, y_test))

#print(classification_report(y_test, y_pred))
# cm = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
#             cbar=False)
# ax.set(xlabel="Pred", ylabel="True", xticklabels=categories,
#        yticklabels=categories, title="Confusion matrix")
# plt.yticks(rotation=0)
# plt.show()
#


