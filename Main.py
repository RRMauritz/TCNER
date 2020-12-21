import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load training data --------------------------------
with open('DBLPTrainset.txt') as f:
    lines = f.readlines()

for i in range(len(lines)):
    pre = lines[i].split()[1:]
    lines[i] = [pre[0], ' '.join(pre[1:])]

# Store in pandas data frame
train = pd.DataFrame(lines, columns=['Label', 'Title'])

# Train and apply a vectorizer to the train texts---------------------------------------
vect = CountVectorizer()
vect.fit(train.Title)
X_train_dtm = vect.transform(train.Title)

# Load test data------------------------------------------------
with open('DBLPTestset.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    pre = lines[i].split()[1:]
    lines[i] = [' '.join(pre[1:])]

test = pd.DataFrame(lines, columns=['Title'])

with open('DBLPTestGroundTruth.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()[1:][0]
test['Label'] = lines

X_test_dtm = vect.transform(test.Title)
nb = MultinomialNB()
nb.fit(X_train_dtm, train.Label)
y_test_pred = nb.predict(X_test_dtm)

cm = metrics.confusion_matrix(test.Label, y_test_pred)
ca = metrics.accuracy_score(test.Label, y_test_pred)
print(cm)

print(ca)