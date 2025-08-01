from ADGNMF import ADG_NMF
import numpy as np
import pandas as pd
import time
from evaluation import eva

csv_file = 'data\\PBMC_RNA_fea_normalized.csv'
df1 = pd.read_csv(csv_file, index_col=0)#X1 is designated for omics data with fewer features.
csv_file = 'data\\PBMC_ATAC_fea_normalized.csv'
df2 = pd.read_csv(csv_file, index_col=0)#X2 is designated for omics data with more features.
label = [i.split('_')[0] for i in df1.columns]
df1.columns = label
df2.columns = label
num_columns =df1.shape[1]
DX = np.eye(num_columns)
S = np.eye(num_columns)
DS = np.eye(num_columns)
#Best values of its hyperparameters A and B are set to 0.01 and 10, respectively.
# A=[0.01]
# B=[10]
A=[0.001,0.01,0.1,1,10,100]
B=[0.001,0.01,0.1,1,10,100]
for a in A:
    for b in B:
        for r in range(20, 21):
            for i in range(1):
                ADGNMF = ADG_NMF(X1=df1, X2=df2, rank=r, DS=S, DX=DX, S=DS,
                                  alpha=a, beta=b, cluster_num=len(np.unique(label)))

                ADGNMF.log_scale()
                ADGNMF.normalize()
                start_time = time.time()
                ADGNMF.snf_integration()
                ADGNMF.LS()
                ADGNMF.factorize()
                end_time = time.time()
                ADGNMF.clustering()

                execution_time = end_time - start_time

                ari, nmi, acc, purity, silhouette_avg, ch_score = eva(label, ADGNMF.cluster, df1.T)
                print('alpha='f'{a}', 'beta='f'{b}', 'rank='f'{r}', "ARI:", ari, "NMI:", nmi, "ACC:", acc, "PURITY:",
                      purity, "silhouette:", silhouette_avg, "ch:", ch_score, "time:", execution_time)
