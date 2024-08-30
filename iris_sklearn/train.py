from pickle import dump

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# import some data to play with
iris = datasets.load_iris()
X = iris.data
Y = iris.target

logreg = LogisticRegression(C=1e5)
logreg.fit(X, Y)


with open("model.pkl", "wb") as f:
    dump(logreg, f, protocol=5)
