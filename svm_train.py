import logging
import logging.config
from joblib import dump
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from lemma_tokenizer import lemma_stopwords, LemmaTokenizer

from consts import TRAIN_DATA
import categories
import datasets
from prepare_data import build_specific_dataset

# Logging
logging.config.fileConfig('logs/conf/logging.conf',
                          defaults={'logfilename': './logs/svm_train.log'})
logger = logging.getLogger('svm_train')

logger.info('Getting training data...')
dataset = build_specific_dataset(TRAIN_DATA, categories.COMP, datasets.comp,
                                 datasets.newsgroups)
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

# na razie random parametry
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

logger.info('Stopwords: %s', lemma_stopwords)
pipeline.set_params(**parameters)

logger.info('Fitting estimator...')
# x_train_counts = vect.fit_transform(newsgroups.data)
# x_train_tf = tfidf.fit_transform(x_train_counts)

logger.info('Fitting classifier...')
# not possible to transform with pipeline. check if necessary

pipeline.fit(dataset.data, dataset.target)
# clf.fit(x_train_tf, newsgroups.target)

logger.info('Saving classifier...')
dump(pipeline, 'clf.joblib')
# dump(clf, 'svm.joblib')

# print("Testing classifier...")
# newsgroups_test = fetch_20newsgroups(subset='test')
#
# # x_test_counts = vect.transform(newsgroups_test.data)
# # x_test_tf = tfidf.transform(x_test_counts)
# y_test = newsgroups_test.target
#
# # no transform
# y_pred = pipeline.predict(x_test)
#
# # y_pred = clf.predict(x_test_tf)
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
