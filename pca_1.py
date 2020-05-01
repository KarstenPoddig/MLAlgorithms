import numpy as np
from sklearn.decomposition import PCA

m = 10000
n = 20

X =  np.random.randn(m*n).reshape(n, m)
B = np.random.exponential(size=n*n).reshape(n,n)
X = np.dot(B, X)
sigma = np.cov(X)
sigma_theoretical = np.dot(B ,B.T)
X = X.T


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)


