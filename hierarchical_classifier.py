import copy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

from bunch import bunch
import categories
from classifier_details import categories_tree, tmp_tree
from consts import TEST_DATA
from datasets import newsgroups
from data_utils import get_examples_filenames
from data_utils import intersect_datasets
from log import getLogger
from others import others
from prepare_data import get_20newsgroups_datasets, set_pos_neg_example
from report import report_category
from svm_classify import classify_dataset

# Logging
logger = getLogger('hierarchical_classifier')


# One category classification process
def classify_one_category(current_category, datasets):
    logger.info('Category %s - classification process started.',
                current_category.category)
    modified_datasets = copy.deepcopy(datasets)

    logger.info('Category %s - marking given examples '
                'as positive or negative.', current_category.category)
    set_pos_neg_example(modified_datasets,
                        current_category.category,
                        current_category.positive_examples,
                        modified_datasets.target_names)

    logger.info('Category %s - classifying given data...',
                current_category.category)
    real_res = classify_dataset(current_category.category,
                                modified_datasets,
                                current_category.classifier_path)

    logger.info('Category %s - intersecting data to'
                ' positive and negative data...',
                current_category.category)
    (negative_data, positive_data) = intersect_datasets(
                                        datasets,
                                        real_res)

    logger.info('Category %s - updating result structure...',
                current_category.category)
    current_category.set_classified_docs(positive_data.filenames)
    target_examples_filenames = get_examples_filenames(modified_datasets,
                                                       1)
    current_category.set_target_classified_docs(target_examples_filenames)
    current_category.set_confusion_matrix(
        confusion_matrix(modified_datasets.target, real_res))
    current_category.set_classification_report(
        classification_report(modified_datasets.target, real_res))
    current_category.set_accuracy_score(
        accuracy_score(modified_datasets.target, real_res))

    logger.info('Category %s - classification process finished.',
                current_category.category)
    return (negative_data, positive_data)


# Hierachical classification process
if __name__ == "__main__":
    logger.info('Hierachical classification process started.')

    current_all_datasets = get_20newsgroups_datasets(TEST_DATA, newsgroups)
    logger.info('All docs count - %d', len(current_all_datasets.target))

    others_cnt = others()
    neg_data = bunch()
    pos_data = bunch()

    for subtree in tmp_tree:
        root_category = subtree[0]
        subcategories = subtree[1]
        (neg_data, pos_data) = classify_one_category(root_category,
                                                     current_all_datasets)

        current_all_datasets = pos_data
        neg_subdata = bunch()
        pos_subdata = bunch()

        # Due to error described in classifier_details
        # if (root_category.category != categories.MISC):
        for subcategory in subcategories:
            (neg_subdata, pos_subdata) = classify_one_category(
                                                    subcategory,
                                                    current_all_datasets)
            current_all_datasets = neg_subdata
        neg_cnt = len(neg_subdata.target)
        others_cnt.sub_others_counts[root_category.category] = neg_cnt

        current_all_datasets = neg_data
    others_cnt.set_main_others_count(len(neg_data.target))
    logger.info('Hierachical classification process finished.')

    logger.info('Summary report:')
    for subtree in tmp_tree:
        root_category = subtree[0]
        subcategories = subtree[1]
        report_category(logger, root_category)
        for subcategory in subcategories:
            report_category(logger, subcategory)

    logger.info('Not classified docs count: \nmain - %d \ncomp - %d '
                '\nrec - %d \nsci - %d \npolitics - %d \nrel - %d '
                '\nmisc - %d', others_cnt.main_others_count,
                others_cnt.sub_others_counts[categories.COMP],
                others_cnt.sub_others_counts[categories.REC],
                others_cnt.sub_others_counts[categories.SCI],
                others_cnt.sub_others_counts[categories.POLITICS],
                others_cnt.sub_others_counts[categories.REL],
                others_cnt.sub_others_counts[categories.MISC])
