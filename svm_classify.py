# uzycie gotowego klasyfikatora do klasyfikacji przykladow

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import fetch_20newsgroups
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from  lemma_tokenizer import lemma_stopwords, LemmaTokenizer


clf = load("clf.joblib")
newsgroups = fetch_20newsgroups(subset='test')

# vect = CountVectorizer(analyzer='word', tokenizer=LemmaTokenizer(), max_features=2000,
#                        stop_words=stopwords.words('english'), max_df=0.3, min_df=0.005, ngram_range=(1,2))
#
# tfidf = TfidfTransformer()
#
#
# x_test_counts = vect.fit_transform(newsgroups_test.data)
# x_test_tf = tfidf.fit_transform(x_test_counts)
# y_test = newsgroups_test.target
#
# y_pred = clf.predict(x_test_tf)
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))

# no estimator fitting

data = newsgroups.data
correct = newsgroups.target

pred = clf.predict(data)

print(confusion_matrix(correct, pred))
print(classification_report(correct, pred))
print(accuracy_score(correct, pred))
