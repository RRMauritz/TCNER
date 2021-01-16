from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def bow_nb(X_train, X_test, y_train):
    """"
    Naive Bayes classifier that works with the bag of words (BOW) model
    """
    nb = Pipeline([('vect', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))), ('clf', MultinomialNB())])
    nb.fit(X_train, y_train)
    return nb.predict(X_test)
