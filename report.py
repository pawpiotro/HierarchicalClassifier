from data_utils import get_categories_examples_count


# Raportowanie wyniku klasyfikacji
def report_category(logger, current_category):
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
