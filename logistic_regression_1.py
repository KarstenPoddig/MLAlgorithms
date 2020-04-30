import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# create some data
X = np.random.randn(10000).reshape(-1, 4) + [1, -2, 3, 5]
X_distorted = X + np.random.randn(10000).reshape(-1, 4)
coeffs = np.random.randn(4).reshape(-1, 1)
z = np.dot(X, coeffs)
quantile = np.percentile(a=z, q=46)
y = (z > quantile).astype('int').reshape(-1)


# build and fit model
clf = LogisticRegression(random_state=0).fit(X_distorted, y)
y_pred = clf.predict(X)
print('Precision: ' + str(precision_score(y, y_pred)))
print('Recall: ' + str(recall_score(y, y_pred)))
print('F1-Score: ' + str(f1_score(y, y_pred)))
print()

# coeffs
print('True Coefficiens: ' + str(coeffs.T))
print('Estimated Coefficiens: ' + str(clf.coef_))
