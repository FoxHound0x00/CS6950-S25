import gudhi as gd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mst_proc import MSTProcessor
import networkx as nx
from tqdm import tqdm
import json
from collections import defaultdict

class PersistenceProcessor:
    def __init__(self, threshold=35):
        self.threshold = threshold
        self.mst_obj = MSTProcessor(threshold=threshold)
            
    def plot_persistence_barcode(persistence, pts=10, ax=None, title=None):
        sns.set_style("whitegrid")

        birth_times = [birth for _, (birth, death) in persistence[:pts]]
        death_times = [death for _, (birth, death) in persistence[:pts]]

        min_birth = min(birth_times)
        max_death = max(d for d in death_times if d < float("inf"))
        delta = (max_death - min_birth) * 0.1
        infinity = max_death + delta
        axis_start = min_birth - delta
        axis_end = max_death + delta * 2

        # fig, ax = plt.subplots(figsize=(8, 5))

        dimensions = sorted(set(dim for dim, _ in persistence[:pts]))
        palette = sns.color_palette("Set1", n_colors=len(dimensions))
        color_map = {dim: palette[i] for i, dim in enumerate(dimensions)}

        for i, (dim, (birth, death)) in enumerate(persistence[:pts]):
            bar_length = (death - birth) if death != float("inf") else (infinity - birth)
            ax.barh(i, bar_length, left=birth, color=color_map[dim], alpha=0.7)

        legend_patches = [mpatches.Patch(color=color_map[dim], label=f"H {dim}") for dim in dimensions]
        ax.legend(handles=legend_patches, loc="best", fontsize=10)

        ax.set_title(f"Persistence Barcode - {title}", fontsize=12)
        ax.set_xlabel("Filtration Value", fontsize=12)
        ax.set_ylabel("Homology Dimension", fontsize=12)
        ax.set_yticks([])
        ax.invert_yaxis()

        if birth_times:
            ax.set_xlim((axis_start, axis_end))
        return ax



    def plot_persistence_diagrams(persistence, pts=10, ax=None, title=None):
        sns.set_style("whitegrid")

        birth_times = [birth for _, (birth, death) in persistence[:pts]]
        death_times = [death for _, (birth, death) in persistence[:pts]]

        min_birth = min(birth_times)
        max_death = max(d for d in death_times if d < float("inf"))

        delta = (max_death - min_birth) * 0.1
        infinity = max_death + 3 * delta
        # infinity = max_death + delta  # Keep infinity separate from max_death
        axis_end = max_death + delta
        axis_start = min_birth - delta

        # fig, ax = plt.subplots(figsize=(6, 6))

        dimensions = sorted(set(dim for dim, _ in persistence[:pts]))
        palette = sns.color_palette("Set1", n_colors=len(dimensions))
        color_map = {dim: palette[i] for i, dim in enumerate(dimensions)}

        x = [birth for (dim, (birth, death)) in persistence[:pts]]
        y = [death if death != float("inf") else infinity for (dim, (birth, death)) in persistence[:pts]]
        c = [color_map[dim] for (dim, (birth, death)) in persistence[:pts]]

        sizes = [20 + 80 * ((death - birth) / (max(1e-5, max_death - min_birth))) for (_, (birth, death)) in persistence[:pts]]
        ax.scatter(x, y, s=sizes, alpha=0.7, color=c, edgecolors="k")
        ax.fill_between(
            [axis_start, axis_end], [axis_start, axis_end], axis_start, color="lightgrey", alpha=0.5
        )

        if any(death == float("inf") for (_, (birth, death)) in persistence[:pts]):
            ax.scatter(
                [min_birth], [infinity], s=150, color="black", marker="*", label="Infinite Death"
            )
            ax.plot([axis_start, axis_end], [infinity, infinity], linewidth=1.0, color="k", alpha=0.6)

            yt = np.array(ax.get_yticks())
            yt = yt[yt < axis_end]  # Avoid out-of-bounds y-ticks
            yt = np.append(yt, infinity)
            ytl = ["%.3f" % e for e in yt]
            ytl[-1] = r"$+\infty$"
            ax.set_yticks(yt)
            ax.set_yticklabels(ytl)

        ax.legend(
            handles=[mpatches.Patch(color=color_map[dim], label=f"H {dim}") for dim in dimensions],
            title="Dimension",
            loc="lower right",
        )

        ax.set_xlabel("Birth", fontsize=12)
        ax.set_ylabel("Death", fontsize=12)
        ax.set_title(f"Persistence Diagram - {title}", fontsize=12)


        ax.set_xlim(axis_start, axis_end)
        ax.set_ylim(min_birth, infinity + delta / 2)

        return ax
    
    def filter(self, persistence_, dists_matrices):
        ### change the number of deaths that are being processed here
        components_ = {k: {} for k in persistence_}
        labels_ = {k: {} for k in persistence_}
        for k,v in persistence_.items():
            persistence = v[:11]
            deaths = [death for _, (birth, death) in persistence[:11]]
            # print(deaths)
            # print(f"persistence - {k}")
            for i in tqdm(range(len(deaths)), desc=f'Processing {k}'):            
                if deaths[i] != np.inf:
                    adj_matrix = (dists_matrices[k] <= deaths[i]).astype(int)
                    # print(adj_matrix)
                    G = nx.from_numpy_array(adj_matrix)
                    conn_comp = list(nx.connected_components(G))
                    # print(f"Number of disconnected components: {len(conn_comp)}")
                    # for idx, component in enumerate(conn_comp, 1):
                    #     print(f"Component {idx}: {component}")

                    mst_obj = MSTProcessor(threshold=deaths[i])
                    n_components, labels, filtered_mst = mst_obj(X=dists_matrices[k], distance_matrix=True)
                    components_[f'{k}'][f'{deaths[i]}'] = n_components
                    labels_[f'{k}'][f'{deaths[i]}'] = labels
                    # print(set(labels))
                    # print(f"{deaths[i]} - {k} - {n_components}")
            # print("--------------------------------")
        
        return components_, labels_
    
    
    def __call__(self, dists_matrices):
        persistence_ = {}
        for k,v in dists_matrices.items():
            # print(f"{k}")
            rips_complex = gd.RipsComplex(distance_matrix=v)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
            persistence = simplex_tree.persistence()
            # print(f"Persistence for {k}: {persistence[:10]}")
            persistence_[f'{k}'] = persistence

        
        
        # for k,v in persistence_.items():
        #     fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        #     self.plot_persistence_barcode(v, pts=10, ax=ax[0], title=k)
        #     self.plot_persistence_diagrams(v, pts=10, ax=ax[1], title=k)
        components_, labels_ = self.filter(persistence_, dists_matrices)
        # print(labels_)
        return persistence_, components_, labels_


