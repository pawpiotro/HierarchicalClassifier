import os
from joblib import dump
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from lemma_tokenizer import lemma_stopwords, lemma_stopwords2
from lemma_tokenizer import LemmaTokenizer, LemmaTokenizer2
from nltk.corpus import stopwords


from consts import TRAIN_DATA, CLFS_FOLDER
import classifier_details
from prepare_data import build_specific_dataset
from log import getLogger

# Logowanie
logger = getLogger('svm_train')


# Metoda do trenowania klasyfikatora danej kategorii
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

    # Parametry dla klasyfikatora kategorii: comp
    '''
    parameters = {
        'vect__analyzer': 'word',
        'vect__tokenizer': LemmaTokenizer(),
        'vect__max_features': 1000,
        'vect__stop_words': lemma_stopwords,
        # 'vect__stop_words': stopwords.words('english'),
        'vect__max_df': 0.706030873393441,
        'vect__min_df': 0.033970089022543615,
        'vect__ngram_range': (1, 2),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__kernel': 'rbf',
        'clf__gamma': 1.0439975865365008,
        'clf__C': 8.437877639063352,
        'clf__decision_function_shape': 'ovr',
        'clf__cache_size': 1000,
        'clf__max_iter': -1
    }
    '''

    # Parametry dla klasyfikatora kategorii: comp_graphics
    '''
    parameters = {
        'vect__analyzer': 'word',
        'vect__tokenizer': LemmaTokenizer(),
        'vect__max_features': 2000,
        'vect__stop_words': lemma_stopwords,
        # 'vect__stop_words': stopwords.words('english'),
        'vect__max_df': 0.6191434298387908,
        'vect__min_df': 0.09695908809214096,
        'vect__ngram_range': (1, 2),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__kernel': 'rbf',
        'clf__gamma': 2.9770837142702145,
        'clf__C': 234.71519573362391,
        'clf__decision_function_shape': 'ovr',
        'clf__cache_size': 1000,
        'clf__max_iter': -1
    }
    '''
    # Parametry dla klasyfikatora kategorii: comp_windows
    '''
    parameters = {
        'vect__analyzer': 'word',
        'vect__tokenizer': LemmaTokenizer(),
        'vect__max_features': 1000,
        'vect__stop_words': lemma_stopwords,
        # 'vect__stop_words': stopwords.words('english'),
        'vect__max_df': 0.7278465425357497,
        'vect__min_df': 0.0051070837987080234,
        'vect__ngram_range': (1, 2),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__kernel': 'rbf',
        'clf__gamma': 0.1465772704812669,
        'clf__C': 125.77150431549414,
        'clf__decision_function_shape': 'ovr',
        'clf__cache_size': 1000,
        'clf__max_iter': -1
    }
    '''

    # Parametry dla klasyfikatora kategorii: comp_ibm
    '''
    parameters = {
        'vect__analyzer': 'word',
        'vect__tokenizer': LemmaTokenizer(),
        'vect__max_features': 1000,
        'vect__stop_words': lemma_stopwords,
        # 'vect__stop_words': stopwords.words('english'),
        'vect__max_df': 0.712781037819612,
        'vect__min_df': 0.07506840606202485,
        'vect__ngram_range': (1, 2),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__kernel': 'rbf',
        'clf__gamma': 2.13379481350056,
        'clf__C': 156.787010880123,
        'clf__decision_function_shape': 'ovr',
        'clf__cache_size': 1000,
        'clf__max_iter': -1
    }
    '''

    # Parametry dla klasyfikatora kategorii: comp_mac
    '''
    parameters = {
        'vect__analyzer': 'word',
        'vect__tokenizer': LemmaTokenizer(),
        'vect__max_features': 2000,
        'vect__stop_words': lemma_stopwords,
        # 'vect__stop_words': stopwords.words('english'),
        'vect__max_df': 0.6708362135295355,
        'vect__min_df': 0.003744710441445909,
        'vect__ngram_range': (1, 2),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__kernel': 'rbf',
        'clf__gamma': 0.5815247858361874,
        'clf__C': 1097.5807898700969,
        'clf__decision_function_shape': 'ovr',
        'clf__cache_size': 1000,
        'clf__max_iter': -1
    }
    '''

    # Parametry dla klasyfikatora kategorii: comp_x
    parameters = {
        'vect__analyzer': 'word',
        'vect__tokenizer': LemmaTokenizer(),
        'vect__max_features': 1000,
        'vect__stop_words': lemma_stopwords,
        # 'vect__stop_words': stopwords.words('english'),
        'vect__max_df': 0.6819413863406742,
        'vect__min_df': 0.027866883704999647,
        'vect__ngram_range': (1, 2),
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__kernel': 'rbf',
        'clf__gamma': 0.22988952016400463,
        'clf__C': 771.3596815378795,
        'clf__decision_function_shape': 'ovr',
        'clf__cache_size': 1000,
        'clf__max_iter': -1
    }

    '''
        'vect__analyzer': 'word',
        'vect__tokenizer': LemmaTokenizer(),
        'vect__max_features': 200,
        parameters = {
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
    '''

    # logger.info('Stopwords: %s', lemma_stopwords)
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
    # Uruchomienie procesu trenowania klasyfikatorów dla podanych kategorii
    categories = [classifier_details.comp_x_details]
    #categories = classifier_details.all_clfs_details
    for clf_details in categories:
        train(clf_details.category,
              clf_details.positive_examples,
              clf_details.all_examples,
              clf_details.classifier_path)
