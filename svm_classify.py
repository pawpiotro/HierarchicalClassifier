from joblib import load
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

clf = load("clf.joblib")

print("Testing classifier...")
newsgroups_test = fetch_20newsgroups(subset='test')
x_test = newsgroups_test.data
y_test = newsgroups_test.target

y_pred = clf.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
