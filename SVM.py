from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer



def svm(X_train, X_test, y_train):
    """"
    Support Vector Machine model
    """
    sgd = Pipeline([('vect', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)
    return sgd.predict(X_test)
