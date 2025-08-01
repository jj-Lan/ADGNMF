# ADGNMF
The source code and input data of ADGNMF.

## Usage
#### Environmental requirements
cupy(download according to your CUDA version), 
numpy, 
scipy, 
scikit-learn, 
pandas


#### Code structure
- ```sc_jnmf.py```: Python script for integrateing single-cell multi-source data via joint NMF and SNF.
- ```_joint_nmf_gpu.py```:  Python script for integrating clustering to update matrices and optimize objectives.
- ```run_RNA-RNA.py```: Python execution script for dimensionality reduction and clustering of multi-source transcriptomic dataset.
- ```run_RNA-ATAC_ADT.py```: Python execution script for dimensionality reduction and clustering of transcriptomic-epigenomic or  dataset.

#### Data Input
In ADGNMF, X1 is designated for omic data with fewer features, and X2 for omic data with more features.

#### Example command
Take the datasets like "PBMC-10K" or "Pollen" as an example.

Running environment: Python version 3.9

Run the model:
```python run_RNA-ATAC.py``` or ```python run_RNA-RNA_ADT.py```


