from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t)
                for t in word_tokenize(doc)]


# test, probably much slower
class LemmaTokenizer2(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t)
                for t in word_tokenize(doc)
                if t not in stopwords.words('english')]


lemma_stopwords = [WordNetLemmatizer().lemmatize(t) for t in stopwords.words('english')]
