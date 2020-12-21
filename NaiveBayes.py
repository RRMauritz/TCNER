from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from Dataload import train, test
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# Compound classifier: vectorizer -> transformer -> classifier
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
               ])
nb.fit(train.Title, train.Label)
y_pred = nb.predict(test.Title)

print('accuracy %s' % accuracy_score(y_pred, test.Label))
print(classification_report(test.Label, y_pred))


