# uzycie gotowego klasyfikatora do klasyfikacji przykladow

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import fetch_20newsgroups
from joblib import dump, load


clf = load("clf.joblib")
newsgroups = fetch_20newsgroups(subset='test')

data = newsgroups.data
correct = newsgroups.target

pred = clf.predict(data)

print(confusion_matrix(correct, pred))
print(classification_report(correct, pred))
print(accuracy_score(correct, pred))
