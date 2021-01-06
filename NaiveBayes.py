from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def bow_nb(X_train, X_test, y_train):
    """"
    Naive Bayes classifier that works with the bag of words (BOW) model
    """
    nb = Pipeline([('vect', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))), ('clf', MultinomialNB())])
    # TODO: uitzoeken of we de max_features kunnen aanpassen en wat ngram_range doet
    # TODO: mogelijk om 'vocabulary = ...' mee te geven aan de TfidfVectorizer (e.g. Chi-Squared)
    nb.fit(X_train, y_train)
    return nb.predict(X_test)
