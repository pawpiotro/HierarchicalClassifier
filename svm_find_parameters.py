# szukanie najlepszych parametrow klasyfikatora

from __future__ import print_function

from pprint import pprint
from time import time

import numpy
from scipy import stats
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from lemma_tokenizer import lemma_stopwords, LemmaTokenizer

data = fetch_20newsgroups(subset='train')

vect = CountVectorizer(analyzer='word', tokenizer=LemmaTokenizer(),
                       stop_words=lemma_stopwords, ngram_range=(1, 2))

tfidf = TfidfTransformer()

clf = svm.SVC(kernel='rbf', decision_function_shape='ovr', cache_size=1000, max_iter=-1)

pipeline = Pipeline([
    ('vect', vect),
    ('tfidf', tfidf),
    ('clf', clf)
])

# Params for gridsearch
parameters = [
    {
        'vect__max_features': [1000, 2000],
        'vect__max_df': [0.3, 0.5, 0.7],
        'vect__min_df': [0.005, 0.01],
        # 'vect__ngram_range': (1, 2),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__gamma': numpy.logspace(-1, 2, 4),
        'clf__C': numpy.logspace(-1, 2, 3),
        # 'clf__decision_function_shape': ['ovr'],
        # 'clf__cache_size': [1000],
        # 'clf__max_iter': [-1],

    }
]

# Params for randomized search
parameters2 = [
    {
        'vect__max_features': [1000, 2000],
        'vect__max_df': stats.uniform(0.5, scale=0.2),
        'vect__min_df': stats.uniform(0, scale=0.1),
        # 'vect__ngram_range': (1, 2),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__gamma': stats.expon(0, 30),
        'clf__C': stats.expon(0, 300),
        # 'clf__decision_function_shape': ['ovr'],
        # 'clf__cache_size': [1000],
        # 'clf__max_iter': [-1],

    }
]

if __name__ == "__main__":
    # grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=20, scoring='f1_macro')
    grid_search = RandomizedSearchCV(pipeline, parameters2[0], cv=5, n_jobs=-1, verbose=20, scoring='f1_macro', n_iter=100)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters2)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()

    print("Best parameters")
    pprint(best_parameters)

    print("Results")
    pprint(grid_search.cv_results_)
    print("Best estimator")
    pprint(grid_search.best_estimator_)

    x_data = numpy.array(data.data)
    y_true = numpy.array(data.target)

    pred = pipeline.predict(x_data)

    print(confusion_matrix(y_true, pred))
    print(classification_report(y_true, pred))
    print(accuracy_score(y_true, pred))
