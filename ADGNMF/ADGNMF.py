import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import Normalizer
from ._joint_nmf_gpu import Joint_NMF_GPU
from .SparseNMF import SparseNMF,init_nmf,nmf_SparseSVD
from sklearn.metrics.pairwise import euclidean_distances


class ADG_NMF:


    def __init__(self, X1, X2, DS, DX, S, rank, alpha=1., beta=1., W1=None, W2=None, H=None,V=None,
                 cluster=None,cluster_num=None):
        self.X1 = X1.astype(np.float32)
        self.X2 = X2.astype(np.float32)
        self.W1 = W1
        self.W2 = W2
        self.H = H
        self.DS = DS
        self.V = V
        self.DX = DX
        self.S = S
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.cluster = cluster
        self.cluster_num = cluster_num

    def gene_selection(self, rm_value1=2, rm_value2=0, threshold=0.06):

        self.gene_set1 = list(set(self.X1.index[(self.X1 > rm_value1).sum(axis=1) > threshold * len(self.X1.columns)])
                              & set(self.X1.index[(self.X1 > rm_value2).sum(axis=1) < (1 - threshold) * len(self.X1.columns)]))
        self.gene_set2 = list(set(self.X2.index[(self.X2 > rm_value1).sum(axis=1) > threshold * len(self.X2.columns)])
                              & set(self.X2.index[(self.X2 > rm_value2).sum(axis=1) < (1 - threshold) * len(self.X2.columns)]))
        self.X1 = self.X1.loc[self.gene_set1, :]
        self.X2 = self.X2.loc[self.gene_set2, :]

    def log_scale(self):

        self.X1 = np.log2(self.X1 + 1)
        self.X2 = np.log2(self.X2 + 1)

    def normalize(self, norm='l1', normalize='cell'):

        norm = Normalizer(norm=norm, copy=False)
        if normalize == 'cell':
            self.X1 = norm.fit_transform(self.X1)
            self.X2 = norm.fit_transform(self.X2)
        elif normalize == 'gene':
            self.X1 = norm.fit_transform(self.X1.T).T
            self.X2 = norm.fit_transform(self.X2.T).T

    def __compute_affinity_matrix(self, X, k=20, sigma=None):
        dists = euclidean_distances(X)
        if sigma is None:
            sigma = np.median(dists)
        A = np.exp(-dists ** 2 / (2 * sigma ** 2))

        for i in range(A.shape[0]):
            idx = np.argsort(A[i, :])[:-k - 1]
            A[i, idx] = 0
            A[idx, i] = 0
        A = (A + A.T) / 2
        row_sums = A.sum(axis=1, keepdims=True)
        A = A / np.maximum(row_sums, 1e-8)
        return A

    def snf_integration(self, k=20, t=20, alpha=0.5, sigma=None):

        H1 = nmf_SparseSVD(X=self.X1, n_components=self.rank, max_iter=500)
        H2 = nmf_SparseSVD(X=self.X2, n_components=self.rank, max_iter=500)

        A1 = self.__compute_affinity_matrix(H1.T, k=k, sigma=sigma)
        A2 = self.__compute_affinity_matrix(H2.T, k=k, sigma=sigma)
        affinity_matrices = [A1, A2]

        n_samples = A1.shape[0]
        P = [np.copy(mat) for mat in affinity_matrices]

        for _ in range(t):
            new_P = []
            for i in range(len(affinity_matrices)):
                contribution = np.zeros((n_samples, n_samples))
                for j in range(len(affinity_matrices)):
                    if i != j:
                        contribution += P[j]
                contribution /= (len(affinity_matrices) - 1) if len(affinity_matrices) > 1 else 1

                diffused = np.dot(affinity_matrices[i], np.dot(contribution, affinity_matrices[i].T))
                new_P_i = alpha * diffused + (1 - alpha) * affinity_matrices[i]
                new_P.append(new_P_i)

            P = new_P

        V = np.mean(P, axis=0)
        self.V = (V + V.T) / 2
        self.DX = np.diag(self.V.sum(axis=1))

    def LS(self):

        EM = nmf_SparseSVD(X=self.X1, n_components=self.rank, max_iter=500)
        cluster = linkage(EM.T, method='ward')
        clusters = fcluster(cluster,
                            t=self.cluster_num,
                            criterion="maxclust")

        n_cells = len(clusters)
        similarity_matrix = np.zeros((n_cells, n_cells), dtype=int)


        for i in range(n_cells):
            for j in range(n_cells):
                if clusters[i] == clusters[j]:
                    similarity_matrix[i, j] = 1



        self.S = similarity_matrix
        self.DS = np.diag(self.S.sum(axis=1))




    def factorize(self, solver='cd', init='SpareSVD'):

        if solver == 'cd':
            if init == 'SpareSVD':
                self.D = np.vstack((self.X1, self.X2))
                len = self.X1.shape[0]
                self.W1, self.W2, self.H = init_nmf(
                    X=self.D, n_components=self.rank, len=len)
            elif self.W1 is None or self.W2 is None or self.H is None:
                print("select 'random' or set the value of factorized matrix.")

            j_nmf = Joint_NMF_GPU(
                self.X1, self.X2, self.W1, self.W2, self.H, self.V, self.S, self.DS, self.DX,
                self.alpha, self.beta, self.cluster_num,
                iter_num=500, conv_judge=1e-5, calc_log=[])
            self.W1, self.W2, self.H , self.S, self.V, self.DS, self.DX= j_nmf.calc()


    def clustering(self):
        self.cluster = linkage(self.H.T, method='ward')
        self.cluster = fcluster(self.cluster,
                                t=self.cluster_num,
                                criterion="maxclust")

