from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from hookfn import Probe
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTForImageClassification.from_pretrained('/workspace/ixai_stuff/notebooks/vit_cifar10_finetuned')
model.to(device)

# load cifar10 (only small portion for demonstration purposes) 
train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}


processor = ViTImageProcessor.from_pretrained("/workspace/ixai_stuff/notebooks/vit_cifar10_finetuned")
inputs = processor(images=test_ds[:]['img'], return_tensors="pt").to(device)
layer_names = [name for name, _ in model.named_modules() if "intermediate_act_fn" in name]
hook_manager = Probe(model)
hook_manager.register_hook(layers_=layer_names)
activations = hook_manager.layer_outs
json.dump(activations, open('activations.json', 'w'), cls=NumpyEncoder)

with torch.no_grad():
  outputs = model(inputs.pixel_values)
logits = outputs.logits
logits.shape

y_true = test_ds[:]['label']
y_pred = outputs.logits.argmax(1).cpu().numpy()

labels = [i for i in label2id.keys()]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
plt.gcf().savefig('confusion_matrix.png', dpi=1200, bbox_inches='tight')

## some viz
from act_viz import ActViz
viz = ActViz(id2label)
viz.visualize_activation(activations[list(activations)[-1]], y_true, id2label, method='PCA')
viz.visualize_activation(activations[list(activations)[-1]], y_true, id2label, method='T-SNE')
viz.plot_all(activations, y_true, id2label)
act_list = np.array([ activations[i].detach().cpu().numpy() for i in activations.keys()])
pcs_ = [act_list[i][:, 0, :] for i in range(act_list.shape[0])]
pcs_ = np.array(pcs_)


#### should I do PCA or not!!!!!? Too slow in real-time if I want to do it on a full-rank matrix

for i, pc in enumerate(pcs_):
    os.makedirs(f'layer_{i}/', exist_ok=True)
    dists = distance_matrix(pc)
    mst_obj = MSTProcessor(threshold=5)
    n_components, labels, filtered_mst = mst_obj(pc)
    # print(n_components) --> very misleading, there are too many outliers in real data.
    
    # not realistic to calculate mahalanobis distance for a full-rank matrix. 
    X_pca = mst_obj.pca_utils(pc, n_components=10)
    X_pca_save = mst_obj.pca_utils(pc, n_components=2)
    np.save(f'layer_{i}/X_pca.npy', X_pca_save)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='inferno')
    plt.savefig(f'layer_{i}/pca.png', dpi=1200)
    
    maha_dist = mst_obj.fast_maha(X_pca)
    den_dists_maha = mst_obj.density_normalizer(X=X_pca, dists=maha_dist, k=5)
    den_dists_euclid = mst_obj.density_normalizer(X=pc, dists=dists, k=5)

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
    plt.savefig(f'layer_{i}/distances.png', dpi=1200)

    dists_matrices = {
        'maha_dist': maha_dist,
        'euclid_dist': dists,
        'density_maha': den_dists_maha,
        'density_euclid': den_dists_euclid,
    }

    ph_obj = PersistenceProcessor()
    persistence_, components_, labels_ = ph_obj(dists_matrices)
    
    

    json.dump(persistence_, open(f'layer_{i}/persistence.json', 'w'))
    json.dump(components_, open(f'layer_{i}/components.json', 'w'))
    # json.dump(({k: v.tolist() for k, v in labels_.items()}), open('labels.json', 'w'))
    json.dump(labels_, open(f'layer_{i}/labels.json', 'w'), cls=NumpyEncoder)
    
    
    json.dump(y_true, open(f'layer_{i}/y_true.json', 'w'), cls=NumpyEncoder)
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


folders = [f for f in os.listdir('.') if os.path.isdir(f) and f.startswith('layer_')]

for i in range(len(folders)):
    labels_ = json.load(open(f'{folders[i]}/labels.json', 'r'))
    y = json.load(open(f'{folders[i]}/y_true.json', 'r'))
    jsons_ = gen_ph_data(labels_, y)
    json.dump(jsons_, open(f'{folders[i]}/ph_data_all.json', 'w'), cls=NumpyEncoder)
    for k, v in jsons_.items():
        json.dump(v, open(f'{folders[i]}/ph_data_{k}.json', 'w'))