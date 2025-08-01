import cupy as cp
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

class Joint_NMF_GPU(object):
    def __init__(
            self, X1, X2, W1, W2, H, V, S, DS, DX,
            alpha, beta, cluster_num,
            iter_num=100, conv_judge=1e-5, calc_log=[], regularization='l1'):
        self.X1 = cp.asarray(X1)
        self.X2 = cp.asarray(X2)
        self.W1 = cp.asarray(W1)
        self.W2 = cp.asarray(W2)
        self.H = cp.asarray(H)
        self.V = cp.asarray(V)
        self.S = cp.asarray(S)
        self.DS = cp.asarray(DS)
        self.DX = cp.asarray(DX)
        self.alpha = alpha
        self.beta = beta
        self.iter_num = iter_num
        self.conv_judge = conv_judge
        self.calc_log = calc_log
        self.regularization = 'l1'
        self.cluster_num = cluster_num

    def __update_W1(self):
        self.W1 *= cp.divide(2*self.X1.dot(self.H.T) ,
                             self.W1.dot(self.H.dot(self.H.T))+ self.X1.dot(self.X1.T.dot(self.W1)))

    def __update_W2(self):
        self.W2 *= cp.divide(2*self.X2.dot(self.H.T) ,
                             self.W2.dot(self.H.dot(self.H.T))+ self.X2.dot(self.X2.T.dot(self.W2)))

    def __update_H(self):
        self.H *= cp.divide(
            2*self.W1.T.dot(self.X1) + 2*self.W2.T.dot(self.X2)+self.alpha * self.H.dot(self.S)
            +self.beta * self.H.dot(self.V),
            self.alpha * self.H.dot(self.DS) + self.beta * self.H.dot(self.DX)
            + self.W1.T.dot(self.W1.dot(self.H))+ self.W2.T.dot(self.W2.dot(self.H)))


    def __update_Ls(self):
        H_cp = self.H

        H_np = cp.asnumpy(H_cp.T)
        cluster = linkage(H_np, method='ward')
        clusters_np = fcluster(cluster,
                            t=self.cluster_num,
                            criterion="maxclust")

        clusters = cp.asarray(clusters_np)


        n_cells = len(clusters)

        clusters_col = clusters.reshape(-1, 1)
        clusters_row = clusters.reshape(1, -1)
        similarity_matrix = cp.equal(clusters_col, clusters_row).astype(cp.int32)
        self.S = similarity_matrix
        S = similarity_matrix.get()
        row_sums = S.sum(axis=1)
        DS = np.diag(row_sums)
        self.DS=cp.asarray(DS)

    def __calc_min_func(self):

        if self.regularization == 'l1':
            return cp.linalg.norm(self.X1 - self.W1.dot(self.H), ord='fro')**2 \
                + cp.linalg.norm(self.X2 - self.W2.dot(self.H), ord='fro')**2 \
                + cp.linalg.norm(self.H - self.W1.T.dot(self.X1), ord='fro') ** 2 \
                + cp.linalg.norm(self.H - self.W2.T.dot(self.X2), ord='fro')** 2 \
                + self.alpha * cp.trace(self.H.dot(self.DS-self.S).dot(self.H.T)) \
                + self.beta * cp.trace(self.H.dot(self.DX-self.V).dot(self.H.T)) \


    def calc(self):
        pre_min_func = self.__calc_min_func()

        for cnt in range(self.iter_num):
            self.__update_W1()
            self.W1[self.W1 < cp.finfo(self.W1.dtype).eps] = cp.finfo(
                self.W1.dtype).eps

            self.__update_W2()
            self.W2[self.W2 < cp.finfo(self.W2.dtype).eps] = cp.finfo(
                self.W2.dtype).eps

            self.__update_H()
            self.H[self.H < cp.finfo(self.H.dtype).eps] = cp.finfo(
                self.H.dtype).eps
            self.__update_Ls()


            min_func = self.__calc_min_func()
            self.calc_log.append(min_func)
            pre_min_func = min_func
        return cp.asnumpy(self.W1), cp.asnumpy(self.W2), cp.asnumpy(self.H), cp.asnumpy(self.V), cp.asnumpy(self.S), cp.asnumpy(self.DS), cp.asnumpy(self.DX)
