from sklearn.datasets import make_classification, make_blobs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mst_proc import MSTProcessor
from ph import PersistenceProcessor
import json
from collections import defaultdict

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)
    
def euclidean(a: np.ndarray,b: np.ndarray)->  np.ndarray:
    diff = a - b
    ssd = np.sum(diff**2, axis=1)
    return np.sqrt(ssd)

def distance_matrix(x: np.ndarray) -> np.ndarray:
    pts, dims = x.shape
    i, j = np.triu_indices(pts, k=1)            # Upper Traingular index Without Diagonal index
    a = x[i]                                    # Selecting elements for upper triangular distance computation
    b = x[j]                                    # Selecting elements for upper triangular distance computation
    upper_triangle_distance =  euclidean(a,b)
    d_mat = np.zeros((pts, pts))                # Distance Matrix with all 0
    d_mat[i,j] = upper_triangle_distance        # Filling Up Upper Triangular Matrix
    d_mat = d_mat + d_mat.T                     # Filling Up lower Triangular Matrix
    return d_mat


X, y = make_blobs(n_samples=250, centers=5, n_features=25, random_state=42)
pca = PCA(n_components=2)
pca.fit(X)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='inferno')
plt.savefig('./pca_blobs.png', dpi=1200)

dists = distance_matrix(X)
mst_obj = MSTProcessor(threshold=5)
n_components, labels, filtered_mst = mst_obj(X)
# print(n_components)

X_pca = mst_obj.pca_utils(X, n_components=10)
maha_dist = mst_obj.fast_maha(X_pca)
den_dists_maha = mst_obj.density_normalizer(X=X_pca, dists=maha_dist, k=5)
den_dists_euclid = mst_obj.density_normalizer(X=X, dists=dists, k=5)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
sns.heatmap(den_dists_maha, ax=axes[0,0], cmap='coolwarm', cbar_kws={'label': 'Colorbar 1'})
axes[0,0].set_title("Density Normalized - Mahalanobis Distance")
sns.heatmap(den_dists_euclid, ax=axes[0,1], cmap='coolwarm', cbar_kws={'label': 'Colorbar 2'})
axes[0,1].set_title("Density Normalized - Euclidean Distance")
sns.heatmap(maha_dist, ax=axes[1,0], cmap='coolwarm', cbar_kws={'label': 'Colorbar 3'})
axes[1,0].set_title("Mahalanobis Distance")
sns.heatmap(dists, ax=axes[1,1], cmap='coolwarm', cbar_kws={'label': 'Colorbar 4'})
axes[1,1].set_title("Euclidean Distance")
plt.tight_layout()
plt.savefig('./distances.png', dpi=1200)

dists_matrices = {
    'maha_dist': maha_dist,
    'euclid_dist': dists,
    'density_maha': den_dists_maha,
    'density_euclid': den_dists_euclid,
}

ph_obj = PersistenceProcessor()
persistence_, components_, labels_ = ph_obj(dists_matrices)

json.dump(persistence_, open('persistence.json', 'w'))
json.dump(components_, open('components.json', 'w'))
# json.dump(({k: v.tolist() for k, v in labels_.items()}), open('labels.json', 'w'))
json.dump(labels_, open('labels.json', 'w'), cls=NumpyEncoder)
json.dump(y, open('y.json', 'w'), cls=NumpyEncoder)
# np.save('labels.npy', y)

import pandas as pd
def gen_ph_data(labels_, y_true, output="sankey_data", threshold=None):
    jsons_ = {}
    for metric, deaths in labels_.items():
        jsons_[metric] = {}
        jsons_[metric]["Original Labels"] = list(map(int, y_true))
        
        for death_value, values in deaths.items():
            # print(f"death_value: {death_value}, values: {values}")
            if threshold is not None and int(death_value) >= threshold:
                continue
            death_key = str(death_value)
            jsons_[metric][death_key] = list(map(int, values))

            
        
    return jsons_


    
jsons_ = gen_ph_data(labels_, y)
json.dump(jsons_, open('ph_data_all.json', 'w'), cls=NumpyEncoder)

for k, v in jsons_.items():
    json.dump(v, open(f'ph_data_{k}.json', 'w'))