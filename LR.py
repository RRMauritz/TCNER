from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def logrec(X_train, X_test, y_train):
    """"
    Logistic Regression model
    """

    logreg = Pipeline([('vect', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5, multi_class='auto', solver='lbfgs')),
                       ])
    logreg.fit(X_train, y_train)
    return logreg.predict(X_test)
