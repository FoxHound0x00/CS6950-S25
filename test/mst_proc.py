import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import networkx as nx


class MSTProcessor:
    def __init__(self, threshold=35):
        np.random.seed(42)  # for reproducibility
        self.threshold = threshold

    def create_mst(self, X, distance_matrix=False):
        dist_matrix = X
        if not distance_matrix:
            dist_matrix = squareform(pdist(X))
        mst = minimum_spanning_tree(dist_matrix)
        return mst.toarray()

    def filter_mst(self, mst, threshold):
        mst[mst > threshold] = 0
        return mst

    def fast_maha(self, X, n_comps=50):
        D = pdist(X, metric='mahalanobis', VI=None)
        dist_maha = squareform(D)
        return dist_maha

    def ncomps_(self, mst):
        return connected_components(mst, directed=False)

    def pca_utils(self, X, n_components=50):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return X_pca

    def __call__(self, X, distance_matrix=False):
        mst_dense = self.create_mst(X, distance_matrix)
        # print(mst_dense)
        filtered_mst = self.filter_mst(mst_dense, self.threshold)
        # print(filtered_mst.shape)
        n_components, labels = self.ncomps_(filtered_mst)
        return n_components, labels, filtered_mst

    # perform time profiling
    # def density_normalizer(self, X, dists, k):
    #     sorted_idx = np.argsort(dists, axis=1)
    #     kth_ngbr = sorted_idx[:, k-1]
    #     kth_dist = dists[np.arange(dists.shape[0]), kth_ngbr]
    #     norm_dist = dists / kth_dist[:, np.newaxis]
    #     return norm_dist

    def density_normalizer(self, X, dists, k):
        ## fixed, symmetric
        triu_indices = np.triu_indices_from(dists, k=1)
        sorted_idx = np.argsort(dists, axis=1)
        kth_ngbr = sorted_idx[:, k-1]
        kth_dist = dists[np.arange(dists.shape[0]), kth_ngbr]
        norm_dist = np.zeros_like(dists)
        norm_dist[triu_indices] = dists[triu_indices] / kth_dist[triu_indices[0]]
        norm_dist += norm_dist.T
        return norm_dist

    def plot_stacked_cluster_chart(self, pruned_mst, n_comps, labels, node_classes):
        keys_ = set(labels)
        classes_ = set(node_classes)
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

        plt.show()

        return counts