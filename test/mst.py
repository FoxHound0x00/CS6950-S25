import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import os

class MSTProcessor:
    def __init__(self, threshold=35):
        np.random.seed(42)  # for reproducibility
        self.threshold = threshold

    def create_mst(self, X):
        dist_matrix = squareform(pdist(X))
        mst = minimum_spanning_tree(dist_matrix)
        return mst.toarray()

    def filter_mst(self, mst, threshold):
        mst[mst > threshold] = 0
        return mst

    def ncomps_(self, mst):
        return connected_components(mst, directed=False)

    def __call__(self, X):
        mst_dense = self.create_mst(X)
        filtered_mst = self.filter_mst(mst_dense, self.threshold)
        print(filtered_mst.shape)
        n_components, labels = self.ncomps_(filtered_mst)
        return n_components, labels, filtered_mst

    def plot_stacked_cluster_chart(self, labels, y_true, layer):
        keys_ = set(labels)
        classes_ = set(y_true)
        components = {key: [] for key in keys_}
        value_range = np.arange(len(set(labels)))

        for i in range(len(y_true)):
            components[labels[i]].append(y_true[i])
        
        counts = np.array([[values.count(cls) for cls in classes_] for values in components.values()])
        percentages = counts / counts.sum(axis=1, keepdims=True) * 100
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bottom = np.zeros(len(components))
        
        for i, cls in enumerate(classes_):
            ax.bar(components.keys(), percentages[:, i], label=f'Class {cls}', bottom=bottom)
            bottom += percentages[:, i]

        ax.set_ylabel('Percentage')
        ax.set_title('Stacked Bar Chart of Components')
        ax.legend(title='Classes')

        os.makedirs("SBC", exist_ok=True)
        plt.savefig(f"SBC/Layer_{layer}.png")        
        return counts


# mst_obj = MSTProcessor(threshold=25)
# n_components, labels, filtered_mst = mst_obj(pcs[11])
# print(n_components)

