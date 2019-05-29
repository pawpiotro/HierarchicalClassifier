import logging
import logging.config
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

from consts import TEST_DATA
from prepare_data import build_specific_dataset

# Logging
logging.config.fileConfig('logs/conf/logging.conf',
                          defaults={'logfilename': './logs/svm_classify.log'})
logger = logging.getLogger('svm_classify')


# Classifications methods
def classify(category, positive_examples, all_examples, classifier_path):
    logger.info('Loading classifier: %s', classifier_path)
    clf = load(classifier_path)

    logger.info('Getting test data for category: %s...', category)
    dataset = build_specific_dataset(TEST_DATA, category, positive_examples,
                                     all_examples)

    x_test = dataset.data
    y_test = dataset.target

    logger.info('Testing classifier: %s', classifier_path)
    y_pred = clf.predict(x_test)

    logger.info('Results for classifier: %s', classifier_path)
    logger.info(confusion_matrix(y_test, y_pred))
    logger.info(classification_report(y_test, y_pred))
    logger.info(accuracy_score(y_test, y_pred))

    return (y_test, y_pred)


def classify_dataset(category, dataset, classifier_path):
    logger.info('Loading classifier: %s', classifier_path)
    clf = load(classifier_path)

    x_test = dataset.data
    y_test = dataset.target

    logger.info('Testing classifier: %s', classifier_path)
    y_pred = clf.predict(x_test)

    logger.info('Results for classifier: %s', classifier_path)
    logger.info(confusion_matrix(y_test, y_pred))
    logger.info(classification_report(y_test, y_pred))
    logger.info(accuracy_score(y_test, y_pred))

    return (y_test, y_pred)
