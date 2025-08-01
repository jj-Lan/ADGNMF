import numpy as np
from SparseNMF import SparseNMF


class TermDocumentReduce:

    def __init__(self, n_components=25):
        self.n_components = n_components
        self.nmf_cluster = NMFCluster(self.n_components)

    def fit(self, X):
        self.nmf_cluster.fit(X)

    def fit_transform(self, X):
        if self.nmf_cluster.topics_ is None:
            self.fit(X)
        return self.transform(X)

    def transform(self, X):
        assert self.nmf_cluster.topics_ is not None
        x_new = np.zeros((X.shape[0], self.n_components))
        for c in range(self.n_components):
            terms = np.where(self.nmf_cluster.topics_ == c)[0]
            x_new[:, c] = np.linalg.norm(X[:, terms], axis=1)
        return x_new


class NMFCluster:

    def __init__(self, n_clusters=25):
        self.n_clusters = n_clusters
        self.nmf_inst = SparseNMF(n_clusters,  tol=1e-9, max_iter=300)

        self.W = None
        self.H = None
        self.labels_ = None
        self.topics_ = None

    def fit(self, X):
        self.W = self.nmf_inst.fit_transform(X)
        self.H = self.nmf_inst.components_
        self.labels_ = np.argmax(self.W, axis=1)
        self.topics_ = np.argmax(self.H, axis=0)

    def fit_predict(self, X):

        if self.labels_ is None:
            self.fit(X)
        return self.labels_

    def fit_predict_documents(self, X):

        return self.fit_predict(X)

    def fit_predict_terms(self, X):

        if self.topics_ is None:
            self.fit(X)
        return self.topics_

    def predict(self, X):

        return np.argmax(np.dot(X, self.H.T), axis=1)

    def predict_documents(self, X):

        return self.predict(X)

    def predict_terms(self, X):

        return np.argmax(np.dot(X.T, self.W.T), axis=1)
