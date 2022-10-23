import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = load_iris
(X_iris, y_iris) = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state = 0)

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris,random_state=1)   
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

clf.fit(X_train, y_train)

tree.plot_tree(clf.fit(X_train, y_train) )
clf.score(X_test, y_test)
