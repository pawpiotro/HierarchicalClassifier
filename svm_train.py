import numpy
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from joblib import dump, load


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


print("Reading data...")
newsgroups_train = fetch_20newsgroups(subset='train')
computer_train = fetch_20newsgroups(subset='train', categories=['comp.graphics',
                                                                'comp.os.ms-windows.misc',
                                                                'comp.sys.ibm.pc.hardware',
                                                                'comp.sys.mac.hardware',
                                                                'comp.windows.x'])
misc_train = fetch_20newsgroups(subset='train', categories=['misc.forsale'])
recreation_train = fetch_20newsgroups(subset='train', categories=['rec.autos',
                                                                  'rec.motorcycles',
                                                                  'rec.sport.baseball',
                                                                  'rec.sport.hockey'])
science_train = fetch_20newsgroups(subset='train', categories=['sci.crypt',
                                                               'sci.electronics',
                                                               'sci.med',
                                                               'sci.space'])
politics_train = fetch_20newsgroups(subset='train', categories=['talk.politics.misc',
                                                                'talk.politics.guns',
                                                                'talk.politics.mideast'])
religion_train = fetch_20newsgroups(subset='train', categories=['talk.religion.misc',
                                                                'alt.atheism',
                                                                'soc.religion.christian'])

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.SVC()),
])

parameters = {
    'vect__analyzer': 'word',
    'vect__tokenizer': LemmaTokenizer(),
    'vect__max_features': 2000,
    'vect__stop_words': 'english',
    'vect__max_df': 0.3,
    'vect__min_df': 0.005,
    'vect__ngram_range': (1, 2),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__kernel': 'rbf',
    'clf__gamma': 1.2,
    'clf__C': 1.0,
    'clf__decision_function_shape': 'ovr',
    'clf__cache_size': 1000,
    'clf__max_iter': -1
}

print(parameters)
pipeline.set_params(**parameters)

print("Fitting...")
pipeline.fit(newsgroups_train.data, newsgroups_train.target)

print("Saving classifier...")
dump(pipeline, 'clf.joblib')
