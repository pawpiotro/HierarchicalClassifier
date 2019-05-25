# remove=('headers', 'footers', 'quotes') <- add to fetch for more realistic data

comp = ['comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x']
misc = ['misc.forsale']
rec = ['rec.autos',
       'rec.motorcycles',
       'rec.sport.baseball',
       'rec.sport.hockey']
sci = ['sci.crypt',
       'sci.electronics',
       'sci.med',
       'sci.space']
politics = ['talk.politics.misc',
            'talk.politics.guns',
            'talk.politics.mideast']
rel = ['talk.religion.misc',
       'alt.atheism',
       'soc.religion.christian']

newsgroups = ['alt.atheism',
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.mideast',
              'talk.politics.misc',
              'talk.religion.misc']


def build_dataset(name, l1,l2):
    print([x for x in l2 if x in l1])
    print([x for x in l2 if x not in l1])
    print('\n')


build_dataset("comp", comp, newsgroups)
build_dataset("misc", misc, newsgroups)
build_dataset("rel", rel, newsgroups)
build_dataset("politics", politics, newsgroups)
build_dataset("rec", rec, newsgroups)
build_dataset("sci", sci, newsgroups)

build_dataset("comp_graphics", ['comp.graphics'], comp)
build_dataset("comp_windows", ['comp.os.ms-windows.misc'], comp)
build_dataset("comp_ibm", ['comp.sys.ibm.pc.hardware'], comp)
build_dataset("comp_mac", ['comp.sys.mac.hardware'], comp)
build_dataset("comp_x", ['comp.windows.x'], comp)

build_dataset("rec_autos", ['rec.autos'], rec)
build_dataset("rec_motorcycles", ['rec.motorcycles'], rec)
build_dataset("rec_baseball", ['rec.sport.baseball'], rec)
build_dataset("rec_hockey", ['rec.sport.hockey'], rec)

build_dataset("rel_atheism", ['alt.atheism'], rel)
build_dataset("rel_christian", ['soc.religion.christian'], rel)

build_dataset("sci_crypt", ['sci.crypt'], sci)
build_dataset("sci_electronics", ['sci.electronics'], sci)
build_dataset("sci_med", ['sci.med'], sci)
build_dataset("sci_space", ['sci.space'], sci)

build_dataset("politics_guns", ['talk.politics.guns'], politics)
build_dataset("politics_mideast", ['talk.politics.mideast'], politics)

# computer = fetch_20newsgroups(subset='train', categories=['comp.graphics',
#                                                           'comp.os.ms-windows.misc',
#                                                           'comp.sys.ibm.pc.hardware',
#                                                           'comp.sys.mac.hardware',
#                                                           'comp.windows.x'])
# misc = fetch_20newsgroups(subset='train', categories=['misc.forsale'])
# recreation = fetch_20newsgroups(subset='train', categories=['rec.autos',
#                                                             'rec.motorcycles',
#                                                             'rec.sport.baseball',
#                                                             'rec.sport.hockey'])
# science = fetch_20newsgroups(subset='train', categories=['sci.crypt',
#                                                          'sci.electronics',
#                                                          'sci.med',
#                                                          'sci.space'])
# politics = fetch_20newsgroups(subset='train', categories=['talk.politics.misc',
#                                                           'talk.politics.guns',
#                                                           'talk.politics.mideast'])
# religion = fetch_20newsgroups(subset='train', categories=['talk.religion.misc',
#                                                           'alt.atheism',
#                                                           'soc.religion.christian'])
# comp_graphics = fetch_20newsgroups(subset='train', categories=['comp.graphics'])
# comp_windows = fetch_20newsgroups(subset='train', categories=['comp.os.ms-windows.misc'])
# comp_ibm = fetch_20newsgroups(subset='train', categories=['comp.sys.ibm.pc.hardware'])
# comp_mac = fetch_20newsgroups(subset='train', categories=['comp.sys.mac.hardware'])
# comp_x = fetch_20newsgroups(subset='train', categories=['comp.windows.x'])
#
# rec_autos = fetch_20newsgroups(subset='train', categories=['rec.autos'])
# rec_motos = fetch_20newsgroups(subset='train', categories=['rec.motorcycles'])
# rec_baseball = fetch_20newsgroups(subset='train', categories=['rec.sport.baseball'])
# rec_hockey = fetch_20newsgroups(subset='train', categories=['rec.sport.hockey'])
#
# sci_med = fetch_20newsgroups(subset='train', categories=['sci.med'])
# sci_space = fetch_20newsgroups(subset='train', categories=['sci.space'])
# sci_electronics = fetch_20newsgroups(subset='train', categories=['sci.electronics'])
# sci_crypt = fetch_20newsgroups(subset='train', categories=['sci.crypt'])
#
# politics_mideast = fetch_20newsgroups(subset='train', categories=['talk.politics.mideast'])
# politics_guns = fetch_20newsgroups(subset='train', categories=['talk.politics.guns'])
#
# rel_atheism = fetch_20newsgroups(subset='train', categories=['alt.atheism'])
# rel_christian = fetch_20newsgroups(subset='train', categories=['soc.religion.christian'])
#
# category_names = [
#     'computer',
#     'misc',
#     'recreation',
#     'science',
#     'politics',
#     'religion',
#     'comp_graphics',
#     'comp_windows',
#     'comp_ibm',
#     'comp_mac',
#     'comp_x',
#     'rec_autos',
#     'rec_motos',
#     'rec_baseball',
#     'rec_hockey',
#     'sci_med',
#     'sci_space',
#     'sci_electronics',
#     'sci_crypt',
#     'politics_mideast',
#     'politics_guns',
#     'rel_atheism',
#     'rel_christian'
# ]