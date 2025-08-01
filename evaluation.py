import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics.cluster import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, 1

from sklearn.preprocessing import LabelEncoder
def eva(y_true, y_pred, df1, epoch=0, pp=True):
    le = LabelEncoder()
    y_true = le.fit_transform(y_true)
    y_true = y_true.astype(np.int64)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    df1 = np.array(df1)
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    contingency_matrix = confusion_matrix(y_true, y_pred)
    purity =  np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    silhouette_avg = silhouette_score(df1, y_pred)
    ch_score = calinski_harabasz_score(df1, y_pred)
    return  ari, nmi, acc, purity,silhouette_avg,ch_score
