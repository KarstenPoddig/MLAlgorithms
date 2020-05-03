import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

#--------------------------- example 1 ------------------------------------------------

# create some data
X = np.random.randn(20000).reshape(-1, 2) + 5
z = 3*X[:, 0] + (-2)*X[:, 1]
quantile = np.percentile(a=z, q=36)
y = (z > quantile).astype('int')

# Decision Tree

decision_tree = DecisionTreeClassifier(max_depth=2)
decision_tree.fit(X, y)
y_pred = decision_tree.predict(X)

print('Precision: ' + str(precision_score(y, y_pred)))
print('Recall: ' + str(recall_score(y, y_pred)))
print('F1-Score: ' + str(f1_score(y, y_pred)))
print()


# Random Forest

random_forest = RandomForestClassifier(max_depth=2, random_state=0)
random_forest.fit(X, y)
y_pred = random_forest.predict(X)

print('Precision: ' + str(precision_score(y, y_pred)))
print('Recall: ' + str(recall_score(y, y_pred)))
print('F1-Score: ' + str(f1_score(y, y_pred)))
print()

base_estimator = random_forest.base_estimator


#--------------------------- example 2 ------------------------------------------------

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10000, n_features=6, n_informative=3, n_redundant=2, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# decision tree

decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)
y_test_pred = decision_tree.predict(X_test)
print('Precision: ' + str(precision_score(y_test, y_test_pred)))
print('Recall: ' + str(recall_score(y_test, y_test_pred)))
print('F1-Score: ' + str(f1_score(y_test, y_test_pred)))
print()


# random forest

random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)
y_test_pred = random_forest.predict(X_test)
print('Precision: ' + str(precision_score(y_test, y_test_pred)))
print('Recall: ' + str(recall_score(y_test, y_test_pred)))
print('F1-Score: ' + str(f1_score(y_test, y_test_pred)))
print()
