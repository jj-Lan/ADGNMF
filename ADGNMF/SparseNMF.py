import numpy as np
from math import sqrt
from sklearn.decomposition import NMF
from sklearn.utils.extmath import svd_flip, squared_norm


class SparseNMF(NMF):

    def __init__(self, n_components=None, solver='cd',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None,  l1_ratio=0., verbose=0,
                 shuffle=False):
        super(SparseNMF, self).__init__(n_components=n_components, init='custom',
                                        solver=solver, beta_loss=beta_loss, tol=tol,
                                        max_iter=max_iter, random_state=random_state,
                                        l1_ratio=l1_ratio, verbose=verbose,
                                        shuffle=shuffle)

    def fit_transform(self, X, y=None, W=None, H=None):

        W, H = init_nmf1(X, n_components=self.n_components)
        return super(SparseNMF, self).fit_transform(X, W=W, H=H)

    def fit(self, X, y=None, **params):

        self.fit_transform(X, **params)
        return self


def norm(x):

    return sqrt(squared_norm(x))


import cupy as cp
import numpy as np
from scipy.sparse.linalg import svds
from numpy.linalg import norm


def svd_flip(u, v):

    u_np = cp.asnumpy(u)
    v_np = cp.asnumpy(v)


    max_abs_cols = np.argmax(np.abs(u_np), axis=0)
    signs = np.sign(u_np[max_abs_cols, range(u_np.shape[1])])
    u_np *= signs
    v_np *= signs[:, np.newaxis]

    return cp.array(u_np), cp.array(v_np)


def init_nmf(X, n_components, len, eps=1e-6):
    X_np = cp.asnumpy(X)
    U, S, V = svds(X_np, n_components)

    S = S[::-1]
    U, V = svd_flip(cp.array(U[:, ::-1]), cp.array(V[::-1]))
    W, H = cp.zeros(U.shape), cp.zeros(V.shape)

    W[:, 0] = cp.sqrt(S[0]) * cp.abs(U[:, 0])
    H[0, :] = cp.sqrt(S[0]) * cp.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]
        x_p, y_p = cp.maximum(x, 0), cp.maximum(y, 0)
        x_n, y_n = cp.abs(cp.minimum(x, 0)), cp.abs(cp.minimum(y, 0))

        x_p_nrm = cp.linalg.norm(x_p)
        y_p_nrm = cp.linalg.norm(y_p)
        x_n_nrm = cp.linalg.norm(x_n)
        y_n_nrm = cp.linalg.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = cp.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W = cp.maximum(W, eps)
    H = cp.maximum(H, eps)


    W1 = W[:len, :]
    W2 = W[len:, :]

    return W1, W2, H


def init_nmf1(X, n_components, eps=1e-6):

    X_np = cp.asnumpy(X)
    U, S, V = svds(X_np, n_components)

    S = S[::-1]
    U, V = svd_flip(cp.array(U[:, ::-1]), cp.array(V[::-1]))
    W, H = cp.zeros(U.shape), cp.zeros(V.shape)

    W[:, 0] = cp.sqrt(S[0]) * cp.abs(U[:, 0])
    H[0, :] = cp.sqrt(S[0]) * cp.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]
        x_p, y_p = cp.maximum(x, 0), cp.maximum(y, 0)
        x_n, y_n = cp.abs(cp.minimum(x, 0)), cp.abs(cp.minimum(y, 0))

        x_p_nrm = cp.linalg.norm(x_p)
        y_p_nrm = cp.linalg.norm(y_p)
        x_n_nrm = cp.linalg.norm(x_n)
        y_n_nrm = cp.linalg.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = cp.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W = cp.maximum(W, eps)
    H = cp.maximum(H, eps)

    return W, H


def nmf_SparseSVD(X, n_components, max_iter=200, tol=1e-4, eps=1e-6):

    X_cp = cp.array(X)
    n_samples, n_features = X_cp.shape

    W, H = init_nmf1(X_cp, n_components, eps)

    W = cp.maximum(W, eps)
    H = cp.maximum(H, eps)

    prev_err = norm(X - cp.asnumpy(cp.dot(W, H)))

    for n_iter in range(1, max_iter + 1):
        H_num = cp.dot(W.T, X_cp)
        H_den = cp.dot(cp.dot(W.T, W), H) + eps
        H *= H_num / H_den
        W_num = cp.dot(X_cp, H.T)
        W_den = cp.dot(W, cp.dot(H, H.T)) + eps
        W *= W_num / W_den
        W = cp.maximum(W, eps)
        H = cp.maximum(H, eps)

        curr_err = norm(X - cp.asnumpy(cp.dot(W, H)))

        if n_iter % 10 == 0:
            err_change = abs(prev_err - curr_err) / (prev_err + eps)
            if err_change < tol:
                break
            prev_err = curr_err

    reconstruction_err = norm(X - cp.asnumpy(cp.dot(W, H)))

    H_np = cp.asnumpy(H)
    return H_np