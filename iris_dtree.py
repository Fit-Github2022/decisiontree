import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

(X_iris, y_iris) = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state = 0)

from sklearn import tree
from sklearn.tree import plot_tree

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris,random_state=1)   
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

a=clf.fit(X_train, y_train)
st.write(a)

fig=plt.figure(figsize=(10,4))
tree.plot_tree(clf.fit(X_train, y_train))
st.pyplot(fig)
