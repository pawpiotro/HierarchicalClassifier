import os
import codecs
import pickle
from sklearn.datasets import get_data_home, fetch_20newsgroups

from consts import CLF_DATA, TRAIN_DATA, TEST_DATA
from consts import POSITIVE_DATA, NEGATIVE_DATA
from classifier_details import all_clfs_details
from log import getLogger

# Logging
logger = getLogger('prepare_data')


# Method to get 20newsgroups data
def get_20newsgroups_datasets(data_type, categories):
    logger.info('get_20newsgroups_datasets args: data_type - %s, '
                'categories - %s', data_type, categories)
    scikit_learn_data_home_path = get_data_home()
    if os.path.exists(scikit_learn_data_home_path):
        # For more realistic data -> remove=('headers', 'footers', 'quotes')
        datasets = fetch_20newsgroups(data_home=scikit_learn_data_home_path,
                                      subset=data_type,
                                      categories=categories)
                                      # remove=('headers', 'footers', 'quotes'))
        return datasets
    else:
        datasets = fetch_20newsgroups(subset=data_type,
                                      categories=categories)
                                      # remove=('headers', 'footers', 'quotes'))
        return datasets


# Methods to create datasets
def build_specific_dataset(data_type, dataset_name, positive_examples,
                           all_examples):
    logger.info('build_specific_dataset args: data_type - %s, '
                'dataset_name - %s, positive_examples - %s, '
                'all_examples - %s', data_type, dataset_name,
                positive_examples, all_examples)
    scikit_learn_data_home_path = get_data_home()
    data_type_path = os.path.join(scikit_learn_data_home_path, CLF_DATA,
                                  data_type)
    dataset_filename = dataset_name + '.pkz'
    dataset_path = os.path.join(data_type_path, dataset_filename)

    if os.path.exists(dataset_path):
        try:
            logger.info('File %s exists.', dataset_path)

            with open(dataset_path, 'rb') as f:
                compressed_dataset = f.read()
            uncompressed_dataset = codecs.decode(
                compressed_dataset, 'zlib_codec')
            dataset = pickle.loads(uncompressed_dataset)

            return dataset
        except Exception as e:
            logger.error('Dataset loading from compressed'
                         'pickle failed: + %s', e)
    else:
        logger.info('File %s doesn\'t exist - creating process started.',
                    dataset_path)

        if not os.path.exists(data_type_path):
            os.makedirs(data_type_path)

        # For more realistic data -> remove=('headers', 'footers', 'quotes')
        dataset = fetch_20newsgroups(data_home=scikit_learn_data_home_path,
                                     subset=data_type,
                                     categories=all_examples)
                                     # remove=('headers', 'footers', 'quotes'))

        set_pos_neg_example(dataset, dataset_name, positive_examples,
                            all_examples)

        compressed_dataset = codecs.encode(pickle.dumps(dataset), 'zlib_codec')

        with open(dataset_path, 'wb') as f:
            f.write(compressed_dataset)
            logger.info('File %s has been created.', dataset_path)

        return dataset


def build_dataset(dataset_name, positive_examples, all_examples):
    build_specific_dataset(TRAIN_DATA, dataset_name, positive_examples,
                           all_examples)
    logger.info('Complete building training dataset: %s', dataset_name)

    build_specific_dataset(TEST_DATA, dataset_name, positive_examples,
                           all_examples)
    logger.info('Complete building test dataset: %s', dataset_name)


# Methods to modify datasets
def set_pos_neg_example(dataset, category, positive_examples, all_examples):
    dataset.target_names = [NEGATIVE_DATA, POSITIVE_DATA]
    positive_count = 0
    negative_count = 0
    for idx in range(len(dataset.target)):
        if all_examples[dataset.target[idx]] in positive_examples:
            '''
            logger.info('Category - %s: '
                        'Document of category %s has been '
                        'classified as positive.',
                        category,
                        all_examples[dataset.target[idx]])
            '''
            dataset.target[idx] = 1
            positive_count += 1
        else:
            '''
            logger.info('Category - %s: '
                        'Document of category %s has been '
                        'classified as negative.',
                        category,
                        all_examples[dataset.target[idx]])
            '''
            dataset.target[idx] = 0
            negative_count += 1
    logger.info('Category - %s: Positive count - %d, negative_count - %d',
                category, positive_count, negative_count)


if __name__ == "__main__":
    # Creating datasets
    for clf_details in all_clfs_details:
        build_dataset(clf_details.category,
                      clf_details.positive_examples,
                      clf_details.all_examples)
