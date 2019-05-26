from joblib import dump
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from lemma_tokenizer import lemma_stopwords, LemmaTokenizer

print("Reading data...")
# remove=('headers', 'footers', 'quotes') <- add to fetch for more realistic data
newsgroups = fetch_20newsgroups(subset='train')

# vect = CountVectorizer(analyzer='word', tokenizer=LemmaTokenizer(), max_features=2000,
#                        stop_words=lemma_stopwords, max_df=0.3, min_df=0.005, ngram_range=(1,2))
#
# tfidf = TfidfTransformer()
#
# clf = svm.SVC(kernel='rbf', gamma=1.2, C=1.0, decision_function_shape='ovr', cache_size=1000, max_iter=-1)


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.SVC())
])

parameters = {
    'vect__analyzer': 'word',
    'vect__tokenizer': LemmaTokenizer(),
    'vect__max_features': 2000,
    'vect__stop_words': lemma_stopwords,
    'vect__max_df': 0.6988043885574027,
    'vect__min_df': 0.009380356271717506,
    'vect__ngram_range': (1, 2),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__kernel': 'rbf',
    'clf__gamma': 2.753097976194963,
    'clf__C': 448.1980411198116,
    'clf__decision_function_shape': 'ovr',
    'clf__cache_size': 1000,
    'clf__max_iter': -1
}

# pprint(list(newsgroups.target_names))
# print(parameters)
print(lemma_stopwords)
pipeline.set_params(**parameters)

print("Fitting classifier...")
pipeline.fit(newsgroups.data, newsgroups.target)

print(newsgroups.data)
print("Saving classifier...")
dump(pipeline, 'clf.joblib')

# print("Testing classifier...")
# newsgroups_test = fetch_20newsgroups(subset='test')
# x_test = newsgroups_test.data
# y_test = newsgroups_test.target
#
# y_pred = pipeline.predict(x_test)
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
