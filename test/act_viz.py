
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class ActViz:
    def __init__(self, id2label):
        self.id2label = id2label


    def plot_act(self, dimred_, labels, id2label, method):
        # plt.figure(figsize=(8, 6))
        # scatter = plt.scatter(dimred_[:, 0], dimred_[:, 1], c=labels, cmap="tab10", s=50)
        # plt.colorbar(scatter, label="Class Labels")
        # plt.title("PCA Visualization")
        # plt.show()
        # class_names = [id2label[label] for label in labels]
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(dimred_[:, 0], dimred_[:, 1], c=labels, cmap="tab10", s=50)
        cbar = plt.colorbar(scatter, label="Class Labels")
        cbar.set_ticks(range(len(id2label)))
        cbar.set_ticklabels([id2label[i] for i in range(len(id2label))])
        plt.title(f"{method} Visualization")
        plt.show()



    def visualize_activation(self, activations, labels, id2label, method='PCA', perplexity=5, n_components=2, random_state=42):
        activations_np = activations.detach().cpu().numpy()
        print(activations_np.shape)
        batch_size, seq_len, hidden_dim = activations_np.shape # [batch_size, seq_len (Sequence length= Number of patches + Class token=196(14*14)+1=197), hidden_dim (768*4=3072)]
        # act_flatten = activations_np.mean(1) # Shape: (batch_size, 3072)
        act_flatten = activations_np[:,0,:]
        print(act_flatten.shape)
        if method == "PCA":
            dimred = PCA(n_components=n_components, random_state=random_state)
        elif method == "T-SNE":
            dimred = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        else:
            raise ValueError("Invalid method. Choose either 'PCA' or 'T-SNE'.")
        
        red_act = dimred.fit_transform(act_flatten)  # Shape: (batch_size * seq_len, n_components)
        self.plot_act(red_act, labels, id2label, method)

    def plot_all(self, activations, y_true, id2label, perplexity=5, n_components=2, random_state=42):
        layers_ = len(activations.keys())

        dimred_pca = []
        dimred_tsne = []
        pca = PCA(n_components=n_components, random_state=random_state)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        for i in activations.keys():
            print(f"Layer: {i}")
            act_np = activations[i].detach().cpu().numpy()
            flat_act = act_np[:,0,:]
            dimred_pca.append(pca.fit_transform(flat_act))
            dimred_tsne.append(tsne.fit_transform(flat_act))
            
        # Plot PCA results
        fig_pca, axes_pca = plt.subplots(1, layers_, figsize=(6 * layers_, 6), sharex=True, sharey=True)
        for i, ax in enumerate(axes_pca):
            scatter = ax.scatter(dimred_pca[i][:, 0], dimred_pca[i][:, 1], c=y_true, cmap="tab10", s=50)
            cbar = fig_pca.colorbar(scatter, ax=ax, label="Class Labels")
            cbar.set_ticks(range(len(id2label)))
            cbar.set_ticklabels([id2label[j] for j in range(len(id2label))])
            ax.set_title(f"layer {i}")
        plt.tight_layout()
        # plt.show()
        plt.savefig('pca_results.png')

        # Plot t-SNE results
        fig_tsne, axes_tsne = plt.subplots(1, layers_, figsize=(6 * layers_, 6), sharex=True, sharey=True)
        for i, ax in enumerate(axes_tsne):
            scatter = ax.scatter(dimred_tsne[i][:, 0], dimred_tsne[i][:, 1], c=y_true, cmap="tab10", s=50)
            cbar = fig_tsne.colorbar(scatter, ax=ax, label="Class Labels")
            cbar.set_ticks(range(len(id2label)))
            cbar.set_ticklabels([id2label[j] for j in range(len(id2label))])
            ax.set_title(f"layer {i}")
        plt.tight_layout()
        # plt.show()
        plt.savefig('tsne_results.png')
