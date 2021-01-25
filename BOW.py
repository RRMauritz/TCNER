from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def bow_nb(X_train, X_test, y_train):
    """"
    Multi-class Naive Bayes classifier that works with the bag of words (BOW) model and TFIDF-ing
    :returns predicted labels for the test data X_test
    """
    nb = Pipeline([('vect', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))), ('clf', MultinomialNB())])
    nb.fit(X_train, y_train)
    return nb.predict(X_test)


def logrec(X_train, X_test, y_train):
    """"
    Multi-class Logistic Regression classifier that works with the bag of words (BOW) model and TFIDF-ing
    :returns predicted labels for the test data X_test
    """

    logreg = Pipeline([('vect', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5, multi_class='auto', solver='lbfgs')),
                       ])
    logreg.fit(X_train, y_train)
    return logreg.predict(X_test)


def svm(X_train, X_test, y_train):
    """"
    Multi-class Support Vector Machine classifier that works with the bag of words (BOW) model and TFIDF-ing
    :returns predicted labels for the test data X_test
    """
    sgd = Pipeline([('vect', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)
    return sgd.predict(X_test)
