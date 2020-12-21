from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from Dataload import train, test

logreg = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                   ])
logreg.fit(train.Title, train.Label)
y_pred = logreg.predict(test.Title)

print('accuracy %s' % accuracy_score(y_pred, test.Label))
print(classification_report(test.Label, y_pred))
