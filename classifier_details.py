import categories
import datasets


class ClassifierDetails:
    def __init__(self, category, positive_examples, all_examples):
        self.category = category
        self.positive_examples = positive_examples
        self.all_examples = all_examples


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

all_clfs_details = [comp_details, misc_details, rel_details, politics_details,
                    rec_details, sci_details, comp_graphics_details,
                    comp_windows_details, comp_ibm_details, comp_mac_details,
                    comp_x_details, rec_autos_details, rec_motorcycles_details,
                    rec_baseball_details, rec_hockey_details,
                    rel_atheism_details, rel_christian_details,
                    sci_crypt_details, sci_electronics_details,
                    sci_med_details, sci_space_details, politics_guns_details,
                    politics_mideast_details]
