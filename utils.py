
import persim 
def compute_dist(x_1, x_2, labels, layers, id2label, dgm1, dgm2, dist_type='bottleneck'):
    print("pass PCA transformed values!!!")
    print("pass h0 or h1 diagrams")
    plt.figure(figsize=(12, 12))
    # diagram 1
    plt.subplot(221)
    scatter = plt.scatter(x_1[:, 0], x_1[:, 1], c=labels, cmap="tab10", s=50)
    cbar = plt.colorbar(scatter, label="Class Labels")
    cbar.set_ticks(range(len(id2label)))
    cbar.set_ticklabels([id2label[i] for i in range(len(id2label))])
    plt.title(f"Point Cloud 1")
    plt.axis("equal")

    # diagram 2
    plt.subplot(222)
    scatter = plt.scatter(x_2[:, 0], x_2[:, 1], c=labels, cmap="tab10", s=50)
    cbar = plt.colorbar(scatter, label="Class Labels")
    cbar.set_ticks(range(len(id2label)))
    cbar.set_ticklabels([id2label[i] for i in range(len(id2label))])
    plt.title(f"Point Cloud 2")
    plt.axis("equal")
    
    # Persistence diagram for x_1
    plt.subplot(223)
    plot_diagrams(dgm1)
    plt.title("Point Cloud 1 Persistence Diagram")
    
    # Persistence diagram for x_2
    plt.subplot(224)
    plot_diagrams(dgm2)
    plt.title("Point Cloud 2 Persistence Diagram")
    
    plt.savefig(f"bottleneck_dist/Layers_{layers[0]}_{layers[1]}_persistence_comparison.png")
    
    
    # computing distances 
    
    plt.figure(figsize=(12, 6))
    plt.subplot(221)
    persim.plot_diagrams([dgm1, dgm2], labels=['Clean $H_0$', 'Noisy $H_0$'], show=False)
    bn_dist, matching = persim.bottleneck(dgm1, dgm2, matching=True)
    
    plt.subplot(222)
    persim.bottleneck_matching(dgm1=dgm1, dgm2=dgm2, matching=matching, labels=['Clean $H_1$', 'Noisy $H_1$'])
    plt.savefig(f"bottleneck_dist/Layers_{layers[0]}_{layers[1]}_distance_comparison.png")
    