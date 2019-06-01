from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

import classifier_details
from data_utils import intersect_datasets, get_examples_filenames
from consts import TEST_DATA
from log import getLogger
from prepare_data import build_specific_dataset
from report import report_category

# Logging
logger = getLogger('svm_classify')


# Classifications methods
def classify(category, positive_examples, all_examples, classifier_path):
    logger.info('Category %s - loading classifier: %s', category,
                classifier_path)
    clf = load(classifier_path)

    logger.info('Category %s - getting test data...', category)
    dataset = build_specific_dataset(TEST_DATA, category, positive_examples,
                                     all_examples)

    x_test = dataset.data
    # y_test = dataset.target

    logger.info('Category %s - testing classifier: %s', category,
                classifier_path)
    y_pred = clf.predict(x_test)

    return (dataset, y_pred)


def classify_dataset(category, dataset, classifier_path):
    logger.info('Category %s - Loading classifier: %s', category,
                classifier_path)
    clf = load(classifier_path)

    x_test = dataset.data
    # y_test = dataset.target

    logger.info('Category %s - Testing classifier: %s', category,
                classifier_path)
    y_pred = clf.predict(x_test)

    return y_pred


if __name__ == "__main__":
    category = classifier_details.comp_details
    (dataset, real_res) = classify(category.category,
                                   category.positive_examples,
                                   category.all_examples,
                                   category.classifier_path)

    logger.info('Category %s - intersecting data to'
                ' positive and negative data...', category.category)
    (negative_data, positive_data) = intersect_datasets(
                                        dataset,
                                        real_res)

    logger.info('Category %s - updating result structure...',
                category.category)
    category.set_classified_docs(positive_data.filenames)
    target_examples_filenames = get_examples_filenames(dataset,
                                                       1)
    category.set_target_classified_docs(target_examples_filenames)
    category.set_confusion_matrix(
        confusion_matrix(dataset.target, real_res))
    category.set_classification_report(
        classification_report(dataset.target, real_res))
    category.set_accuracy_score(
        accuracy_score(dataset.target, real_res))

    logger.info('Summary report:')
    report_category(logger, category)
