import numpy
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from joblib import dump, load

from pprint import pprint

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


print("Reading data...")
# remove=('headers', 'footers', 'quotes') <- add to fetch for more realistic data
newsgroups = fetch_20newsgroups(subset='train')

vect = CountVectorizer(analyzer='word', tokenizer=LemmaTokenizer(), max_features=2000,
                       stop_words=stopwords.words('english'), max_df=0.3, min_df=0.005, ngram_range=(1,2))

tfidf = TfidfTransformer()

clf = svm.SVC(kernel='rbf', gamma=1.2, C=1.0, decision_function_shape='ovr', cache_size=1000, max_iter=-1)
#
# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', svm.SVC())
# ])
#
# # na razie random parametry
# parameters = {
#     'vect__analyzer': 'word',
#     'vect__tokenizer': LemmaTokenizer(),
#     'vect__max_features': 2000,
#     'vect__stop_words': stopwords.words('english'),
#     'vect__max_df': 0.3,
#     'vect__min_df': 0.005,
#     'vect__ngram_range': (1, 2),
#     # 'tfidf__use_idf': (True, False),
#     # 'tfidf__norm': ('l1', 'l2'),
#     'clf__kernel': 'rbf',
#     'clf__gamma': 1.2,
#     'clf__C': 1.0,
#     'clf__decision_function_shape': 'ovr',
#     'clf__cache_size': 1000,
#     'clf__max_iter': -1
# }
#
# pprint(list(newsgroups.target_names))
# print(parameters)
# pipeline.set_params(**parameters)

print("Fitting estimator...")
x_train_counts = vect.fit_transform(newsgroups.data)
x_train_tf = tfidf.fit_transform(x_train_counts)

print("Fitting classifier...")
# pipeline.fit(newsgroups.data, newsgroups.target)
clf.fit(x_train_tf, newsgroups.target)
print("Saving classifier...")
# dump(pipeline, 'clf.joblib')
dump(clf, 'svm.joblib')

print("Testing classifier...")
newsgroups_test = fetch_20newsgroups(subset='test')

x_test_counts = vect.transform(newsgroups_test.data)
x_test_tf = tfidf.transform(x_test_counts)
y_test = newsgroups_test.target

# y_pred = pipeline.predict(x_test)
y_pred = clf.predict(x_test_tf)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
