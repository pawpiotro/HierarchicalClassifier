import os

import categories
from consts import CLFS_FOLDER
import datasets


class ClassifierDetails:
    def __init__(self, category, positive_examples, all_examples):
        self.category = category
        self.positive_examples = positive_examples
        self.all_examples = all_examples
        self.classifier_path = os.path.join(CLFS_FOLDER,
                                            category + '_clf.joblib')
        self.target_classified_docs = []
        self.classified_docs = []
        self.confusion_matrix = None
        self.classification_report = None
        self.accuracy_score = None

    def set_target_classified_docs(self, target_classified_docs):
        self.target_classified_docs = target_classified_docs

    def set_classified_docs(self, classified_docs):
        self.classified_docs = classified_docs

    def set_confusion_matrix(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix

    def set_classification_report(self, classification_report):
        self.classification_report = classification_report

    def set_accuracy_score(self, accuracy_score):
        self.accuracy_score = accuracy_score


comp_details = ClassifierDetails(categories.COMP, datasets.comp,
                                 datasets.newsgroups)
misc_details = ClassifierDetails(categories.MISC, datasets.misc,
                                 datasets.newsgroups)
rel_details = ClassifierDetails(categories.REL, datasets.rel,
                                datasets.newsgroups)
politics_details = ClassifierDetails(categories.POLITICS, datasets.politics,
                                     datasets.newsgroups)
rec_details = ClassifierDetails(categories.REC, datasets.rec,
                                datasets.newsgroups)
sci_details = ClassifierDetails(categories.SCI, datasets.sci,
                                datasets.newsgroups)

comp_graphics_details = ClassifierDetails(categories.COMP_GRAPHICS,
                                          ['comp.graphics'], datasets.comp)
comp_windows_details = ClassifierDetails(categories.COMP_WINDOWS,
                                         ['comp.os.ms-windows.misc'],
                                         datasets.comp)
comp_ibm_details = ClassifierDetails(categories.COMP_IBM,
                                     ['comp.sys.ibm.pc.hardware'],
                                     datasets.comp)
comp_mac_details = ClassifierDetails(categories.COMP_MAC,
                                     ['comp.sys.mac.hardware'],
                                     datasets.comp)
comp_x_details = ClassifierDetails(categories.COMP_X, ['comp.windows.x'],
                                   datasets.comp)

rec_autos_details = ClassifierDetails(categories.REC_AUTOS, ['rec.autos'],
                                      datasets.rec)
rec_motorcycles_details = ClassifierDetails(categories.REC_MOTORCYCLES,
                                            ['rec.motorcycles'],
                                            datasets.rec)
rec_baseball_details = ClassifierDetails(categories.REC_BASEBALL,
                                         ['rec.sport.baseball'],
                                         datasets.rec)
rec_hockey_details = ClassifierDetails(categories.REC_HOCKEY,
                                       ['rec.sport.hockey'],
                                       datasets.rec)

rel_atheism_details = ClassifierDetails(categories.REL_ATHEISM,
                                        ['alt.atheism'],
                                        datasets.rel)
rel_christian_details = ClassifierDetails(categories.REL_CHRISTIAN,
                                          ['soc.religion.christian'],
                                          datasets.rel)

sci_crypt_details = ClassifierDetails(categories.SCI_CRYPT, ['sci.crypt'],
                                      datasets.sci)
sci_electronics_details = ClassifierDetails(categories.SCI_ELECTRONICS,
                                            ['sci.electronics'],
                                            datasets.sci)
sci_med_details = ClassifierDetails(categories.SCI_MED, ['sci.med'],
                                    datasets.sci)
sci_space_details = ClassifierDetails(categories.SCI_SPACE, ['sci.space'],
                                      datasets.sci)

politics_guns_details = ClassifierDetails(categories.POLITICS_GUNS,
                                          ['talk.politics.guns'],
                                          datasets.politics)
politics_mideast_details = ClassifierDetails(categories.POLITICS_MIDEAST,
                                             ['talk.politics.mideast'],
                                             datasets.politics)
politics_misc_details = ClassifierDetails(categories.POLITICS_MISC,
                                          ['talk.politics.misc'],
                                          datasets.politics)

# ValueError: The number of classes has to be greater than one; got 1 class
misc_forsale_details = ClassifierDetails(categories.MISC_FORSALE,
                                         ['misc.forsale'], datasets.newsgroups)

all_clfs_details = [comp_details, misc_details, rel_details, politics_details,
                    rec_details, sci_details, comp_graphics_details,
                    comp_windows_details, comp_ibm_details, comp_mac_details,
                    comp_x_details, rec_autos_details, rec_motorcycles_details,
                    rec_baseball_details, rec_hockey_details,
                    rel_atheism_details, rel_christian_details,
                    sci_crypt_details, sci_electronics_details,
                    sci_med_details, sci_space_details, politics_guns_details,
                    politics_mideast_details, politics_misc_details,
                    misc_forsale_details]

comp_subtree = [comp_details, [comp_graphics_details,
                               comp_windows_details, comp_ibm_details,
                               comp_mac_details]]

rec_subtree = [rec_details, [rec_autos_details,
                             rec_motorcycles_details,
                             rec_baseball_details,
                             rec_hockey_details]]

sci_subtree = [sci_details, [sci_med_details, sci_space_details,
                             sci_crypt_details, sci_electronics_details]]

politics_subtree = [politics_details, [politics_mideast_details,
                                       politics_guns_details,
                                       politics_misc_details]]

rel_subtree = [rel_details, [rel_atheism_details,
                             rel_christian_details]]

misc_subtree = [misc_details, [misc_forsale_details]]

categories_tree = [comp_subtree, rec_subtree, sci_subtree,
                   politics_subtree, rel_subtree, misc_subtree]

# For tests
tmp_clfs_details = [misc_forsale_details]
tmp_tree = [comp_subtree]
