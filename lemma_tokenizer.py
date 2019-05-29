from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet


def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t, get_wordnet_pos(t))
                for t in word_tokenize(doc)]


# test, probably much slower
class LemmaTokenizer2(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t, get_wordnet_pos(t))
                for t in word_tokenize(doc)
                if t not in stopwords.words('english')]


lemma_stopwords = [WordNetLemmatizer().lemmatize(t, get_wordnet_pos(t)) for t in stopwords.words('english')]
