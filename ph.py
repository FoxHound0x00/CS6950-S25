import warnings
warnings.filterwarnings("ignore")

# %load_ext autoreload
# %autoreload 2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import tadasets
from persim import plot_diagrams, bottleneck
from persim import bottleneck_matching
from ripser import ripser



class PersistenceHomology:
    def __init__(self, labels, y_true, id2label):
        self.labels = labels
        self.y_true = y_true
        self.id2label = id2label
    
    def pca_transform(self, data, n_components=2):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)
    
    def ripser_persistence(self, pcs_, dim=2, PCA=False):
        for i, pc in enumerate(pcs_):
            
            if PCA:
                pc = self.pca_transform(pc)
            
            # full-rank
            tic = time.time()
            res_full = ripser(pc)  
            toc = time.time()
            print(f"Elapsed Time Original Point Cloud: {toc - tic:.3g} seconds")
            print(f"Number of connected components (full-rank) (layer-{i}) : {len(res_full['dgms'][0])}")

            # subsampling
            tic = time.time()
            res_sub = ripser(pc, n_perm=400)
            toc = time.time()
            print(f"Elapsed Time Subsampled Point Cloud: {toc - tic:.3g} seconds")
            print(f"Number of connected components (sub-sample) (layer-{i}) : {len(res_sub['dgms'][0])}")
            
            dgms_full = res_full['dgms']
            dgms_sub = res_sub['dgms']
            idx_perm = res_sub['idx_perm']

            # class_names = [id2label[label] for label in labels]
            print(len(idx_perm))
            print(len(self.labels))
            print(len(self.y_true))
            y_bootstrap = [self.y_true[i] for i in idx_perm]
            
            
            # some plots
            plt.figure(figsize=(12, 12))
            # Original point cloud plot
            plt.subplot(221)
            scatter = plt.scatter(pc[:, 0], pc[:, 1], c=self.y_true, cmap="tab10", s=50)
            cbar = plt.colorbar(scatter, label="Class Labels")
            cbar.set_ticks(range(len(self.id2label)))
            cbar.set_ticklabels([self.id2label[i] for i in range(len(self.id2label))])
            plt.title(f"Original Point Cloud ({pc.shape[0]} Points) - Layer {i} - PCA: {PCA}") 
            plt.axis("equal")

            # Subsampled point cloud plot
            plt.subplot(222)
            scatter = plt.scatter(pc[idx_perm, 0], pc[idx_perm, 1], c=y_bootstrap, cmap="tab10", s=50)
            cbar = plt.colorbar(scatter, label="Class Labels")
            cbar.set_ticks(range(len(self.id2label)))
            cbar.set_ticklabels([self.id2label[i] for i in range(len(self.id2label))])
            plt.title(f"Subsampled Cloud ({len(idx_perm)} Points) - Layer {i} - PCA: {PCA}")
            plt.axis("equal")
            
            # Persistence diagram for the original point cloud
            plt.subplot(223)
            plot_diagrams(dgms_full)
            plt.title("Original Point Cloud Persistence Diagram")
            
            # Persistence diagram for the subsampled point cloud
            plt.subplot(224)
            plot_diagrams(dgms_sub)
            plt.title("Subsampled Point Cloud Persistence Diagram")
            
            # plt.show()
            plt.savefig(f"PH/Layer_{i}.png")


        
        return ripser(data, distance_matrix=True, maxdim=dim)['dgms']