import copy
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


def get_examples_filenames(datasets, is_positive, result):
    examples_filenames = []
    for i in range(0, len(result)):
        if result[i] == is_positive:
            examples_filenames.append(datasets.filenames[i])

    return examples_filenames


def get_category_idx(categories, category_name):
    for i in range(0, len(categories)):
        if categories[i] == category_name:
            return i

    return -1


def intersect_datasets(datasets, positive_categories,
                       all_categories, real_res):
    positive_examples = bunch()
    positive_examples.target_names = positive_categories
    negative_examples = bunch()
    negative_examples.target_names = [x for x in all_categories
                                      if x not in positive_categories]

    for i in range(0, len(real_res)):
        if real_res[i] == 1:
            positive_examples.data.append(datasets.data[i])
            idx = get_category_idx(positive_examples.target_names,
                                   datasets.target_names[datasets.target[i]])
            positive_examples.target.append(idx)
            positive_examples.filenames.append(datasets.filenames[i])
        else:
            negative_examples.data.append(datasets.data[i])
            idx = get_category_idx(negative_examples.target_names,
                                   datasets.target_names[datasets.target[i]])
            negative_examples.target.append(idx)
            negative_examples.filenames.append(datasets.filenames[i])

    return (negative_examples, positive_examples)


def report_category(current_category):
    logger.info('Category %s - test result for classifier: %s',
                current_category.category, current_category.classifier_path)
    logger.info('Category %s - real classified examples count - %d, '
                'target classified examples count - %d.',
                current_category.category,
                len(current_category.classified_docs),
                len(current_category.target_classified_docs))
    logger.info('Category %s - real classified examples: %s',
                current_category.category,
                current_category.classified_docs)
    logger.info('Category %s - target classified examples: %s',
                current_category.category,
                current_category.target_classified_docs)
    logger.info('Category %s - confusion matrix: \n%s',
                current_category.category,
                current_category.confusion_matrix)
    logger.info('Category %s - classification report: \n%s',
                current_category.category,
                current_category.classification_report)
    logger.info('Category %s - accuracy score: \n%s',
                current_category.category,
                current_category.accuracy_score)


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
                        current_category.all_examples)

    logger.info('Category %s - classifying given data...',
                current_category.category)
    (target_res, real_res) = classify_dataset(current_category.category,
                                              modified_datasets,
                                              current_category.classifier_path)

    logger.info('Category %s - intersecting data to'
                ' positive and negative data...',
                current_category.category)
    (negative_data, positive_data) = intersect_datasets(
                                        datasets,
                                        current_category.positive_examples,
                                        current_category.all_examples,
                                        real_res)

    logger.info('Category %s - updating result structure...',
                current_category.category)
    current_category.set_classified_docs(positive_data.filenames)
    target_examples_filenames = get_examples_filenames(datasets,
                                                       1,
                                                       target_res)
    current_category.set_target_classified_docs(target_examples_filenames)
    current_category.set_confusion_matrix(confusion_matrix(target_res,
                                                           real_res))
    current_category.set_classification_report(classification_report(
                                                                target_res,
                                                                real_res))
    current_category.set_accuracy_score(accuracy_score(target_res, real_res))

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
        if (root_category.category != categories.MISC):
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
