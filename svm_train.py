import os
from joblib import dump
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from lemma_tokenizer import lemma_stopwords, LemmaTokenizer

from consts import TRAIN_DATA, CLFS_FOLDER
from classifier_details import all_clfs_details
from prepare_data import build_specific_dataset
from log import getLogger
from nltk.corpus import stopwords

# Logging
logger = getLogger('svm_train')


# Method to train specific classifier
def train(category, positive_examples, all_examples, classifier_path):
    if not os.path.exists(CLFS_FOLDER):
        os.makedirs(CLFS_FOLDER)

    logger.info('Getting training data for category: %s...', category)
    dataset = build_specific_dataset(TRAIN_DATA, category, positive_examples,
                                     all_examples)
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
        # 'vect__stop_words': stopwords.words('english'),
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

    logger.info('Fitting classifier: %s...', classifier_path)
    pipeline.fit(dataset.data, dataset.target)

    logger.info('Saving classifier: %s...', classifier_path)
    dump(pipeline, classifier_path)

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


if __name__ == "__main__":
    for clf_details in all_clfs_details:
        train(clf_details.category,
              clf_details.positive_examples,
              clf_details.all_examples,
              clf_details.classifier_path)
