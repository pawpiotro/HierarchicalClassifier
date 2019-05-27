import logging
import logging.config
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

from consts import TEST_DATA
import categories
import datasets
from prepare_data import build_specific_dataset

# Logging
logging.config.fileConfig('logs/conf/logging.conf',
                          defaults={'logfilename': './logs/svm_classify.log'})
logger = logging.getLogger('svm_classify')

logger.info('Loading classifier...')
clf = load("clf.joblib")

logger.info('Getting test data...')
dataset = build_specific_dataset(TEST_DATA, categories.COMP, datasets.comp,
                                 datasets.newsgroups)

x_test = dataset.data
y_test = dataset.target

logger.info('Testing classifier...')
y_pred = clf.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
