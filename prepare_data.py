import os
import codecs
import logging
import logging.config
import pickle
from sklearn.datasets import get_data_home, fetch_20newsgroups

from consts import CLF_DATA, TRAIN_DATA, TEST_DATA
from consts import POSITIVE_DATA, NEGATIVE_DATA
import categories
import datasets

# Logging
logging.config.fileConfig('logs/conf/logging.conf',
                          defaults={'logfilename': './logs/prepare_data.log'})
logger = logging.getLogger('prepare_data')


# Methods to create datasets
def build_specific_dataset(data_type, dataset_name, positive_examples,
                           all_examples):
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
            logging.error('Dataset loading from compressed'
                          'pickle failed: + %s', e)
    else:
        logger.info('File %s doesn\'t exist - creating process started.',
                    dataset_path)

        if not os.path.exists(data_type_path):
            os.makedirs(data_type_path)

        # For more realistic data -> remove=('headers', 'footers', 'quotes')
        dataset = fetch_20newsgroups(data_home=scikit_learn_data_home_path,
                                     subset=data_type,
                                     categories=all_examples,
                                     remove=('headers', 'footers', 'quotes'))

        dataset.target_names = [NEGATIVE_DATA, POSITIVE_DATA]
        for idx in range(len(dataset.target)):
            if all_examples[dataset.target[idx]] in positive_examples:
                dataset.target[idx] = 1
            else:
                dataset.target[idx] = 0

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


# Creating datasets
build_dataset(categories.COMP, datasets.comp, datasets.newsgroups)
build_dataset(categories.MISC, datasets.misc, datasets.newsgroups)
build_dataset(categories.REL, datasets.rel, datasets.newsgroups)
build_dataset(categories.POLITICS, datasets.politics, datasets.newsgroups)
build_dataset(categories.REC, datasets.rec, datasets.newsgroups)
build_dataset(categories.SCI, datasets.sci, datasets.newsgroups)

build_dataset(categories.COMP_GRAPHICS, ['comp.graphics'], datasets.comp)
build_dataset(categories.COMP_WINDOWS,
              ['comp.os.ms-windows.misc'], datasets.comp)
build_dataset(categories.COMP_IBM, ['comp.sys.ibm.pc.hardware'], datasets.comp)
build_dataset(categories.COMP_MAC, ['comp.sys.mac.hardware'], datasets.comp)
build_dataset(categories.COMP_X, ['comp.windows.x'], datasets.comp)

build_dataset(categories.REC_AUTOS, ['rec.autos'], datasets.rec)
build_dataset(categories.REC_MOTORCYCLES, ['rec.motorcycles'], datasets.rec)
build_dataset(categories.REC_BASEBALL, ['rec.sport.baseball'], datasets.rec)
build_dataset(categories.REC_HOCKEY, ['rec.sport.hockey'], datasets.rec)

build_dataset(categories.REL_ATHEISM, ['alt.atheism'], datasets.rel)
build_dataset(categories.REL_CHRISTIAN, ['soc.religion.christian'],
              datasets.rel)

build_dataset(categories.SCI_CRYPT, ['sci.crypt'], datasets.sci)
build_dataset(categories.SCI_ELECTRONICS, ['sci.electronics'], datasets.sci)
build_dataset(categories.SCI_MED, ['sci.med'], datasets.sci)
build_dataset(categories.SCI_SPACE, ['sci.space'], datasets.sci)

build_dataset(categories.POLITICS_GUNS, ['talk.politics.guns'],
              datasets.politics)
build_dataset(categories.POLITICS_MIDEAST, ['talk.politics.mideast'],
              datasets.politics)
