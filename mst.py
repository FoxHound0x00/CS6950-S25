import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components

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
        return connected_components(mst, directed=False)[0]

    def __call__(self, X):
        mst_dense = self.create_mst(X)
        filtered_mst = self.filter_mst(mst_dense, self.threshold)
        ncomps_mst = self.ncomps_(filtered_mst)
        return ncomps_mst


mst_obj = MSTProcessor(threshold=5)
ncomps_mst = mst_obj(pcs_[11])
print(ncomps_mst)