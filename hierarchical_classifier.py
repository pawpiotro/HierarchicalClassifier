import copy
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

from bunch import bunch
import categories
from classifier_details import categories_tree, tmp_tree
from consts import TEST_DATA
from datasets import newsgroups
from log import getLogger
from others import others
from prepare_data import get_20newsgroups_datasets, set_pos_neg_example
from svm_classify import classify_dataset

# Logging
logger = getLogger('hierarchical_classifier')


# Helpers methods
def get_examples_count(result, is_positive):
    count = 0
    for x in result:
        if x == is_positive:
            count += 1

    return count


def get_examples_filenames(dataset, is_positive):
    examples_filenames = []
    for i in range(0, len(dataset.target)):
        if dataset.target[i] == is_positive:
            examples_filenames.append(reduce_filename(dataset.filenames[i]))

    return examples_filenames


def add_example(src_dataset, dst_dataset, example_idx):
    dst_dataset.data.append(src_dataset.data[example_idx])
    dst_dataset.filenames.append(reduce_filename(
                                        src_dataset.filenames[example_idx]))

    target_name = src_dataset.target_names[src_dataset.target[example_idx]]
    if (target_name in dst_dataset.target_names):
        idx = dst_dataset.target_names.index(target_name)
        dst_dataset.target.append(idx)
    else:
        dst_dataset.target_names.append(target_name)
        idx = len(dst_dataset.target_names) - 1
        dst_dataset.target.append(idx)


def intersect_datasets(datasets, real_res):
    positive_examples = bunch()
    negative_examples = bunch()

    for i in range(0, len(real_res)):
        if real_res[i] == 1:
            add_example(datasets, positive_examples, i)
        else:
            add_example(datasets, negative_examples, i)

    return (negative_examples, positive_examples)


def reduce_filename(filename):
    second_to_last_slash_idx = filename.rfind(os.path.sep, 0,
                                              filename.rfind(os.path.sep))
    next_idx = second_to_last_slash_idx + 1
    reduced_filename = filename[next_idx:]

    return reduced_filename


def report_category(current_category):
    logger.info('Category %s - test result for classifier: %s',
                current_category.category, current_category.classifier_path)
    logger.info('Category %s - real classified examples count - %d, '
                'target classified examples count - %d.',
                current_category.category,
                len(current_category.classified_docs),
                len(current_category.target_classified_docs))
    logger.info('Category %s - categories examples counts:\n'
                'real classified examples - %s\n'
                'target classified examples - %s\n',
                current_category.category,
                get_categories_examples_count(
                    current_category.classified_docs),
                get_categories_examples_count(
                    current_category.target_classified_docs))
    '''
    logger.info('Category %s - real classified examples: %s',
                current_category.category,
                current_category.classified_docs)
    logger.info('Category %s - target classified examples: %s',
                current_category.category,
                current_category.target_classified_docs)
    '''
    logger.info('Category %s - confusion matrix: \n%s',
                current_category.category,
                current_category.confusion_matrix)
    logger.info('Category %s - classification report: \n%s',
                current_category.category,
                current_category.classification_report)
    logger.info('Category %s - accuracy score: \n%s',
                current_category.category,
                current_category.accuracy_score)


def get_categories_examples_count(examples):
    res = {}

    for example_filename in examples:
        category = example_filename[0:example_filename.find(os.path.sep)]
        if category in res:
            res[category] += 1
        else:
            res[category] = 1

    return res


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

    for subtree in categories_tree:
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
    for subtree in categories_tree:
        root_category = subtree[0]
        subcategories = subtree[1]
        report_category(root_category)
        for subcategory in subcategories:
            report_category(subcategory)

    logger.info('Not classified docs count: \nmain - %d \ncomp - %d '
                '\nrec - %d \nsci - %d \npolitics - %d \nrel - %d '
                '\nmisc - %d', others_cnt.main_others_count,
                others_cnt.sub_others_counts[categories.COMP],
                others_cnt.sub_others_counts[categories.REC],
                others_cnt.sub_others_counts[categories.SCI],
                others_cnt.sub_others_counts[categories.POLITICS],
                others_cnt.sub_others_counts[categories.REL],
                others_cnt.sub_others_counts[categories.MISC])
