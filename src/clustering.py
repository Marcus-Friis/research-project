import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
from graph_utilities import lcc_excluding_no_content
import leidenalg as la

if __name__ == '__main__':
    # load embeds
    print('loading embeds')
    with open('../data/embeds.json', 'r') as f:
        embeds = json.load(f)
        
    keys = np.array(list(embeds.keys()))
    embeds = np.array(list(embeds.values()))
    idx = np.argsort(keys)
    keys = keys[idx]
    embeds = embeds[idx]
    print('Raw shapes', keys.size, embeds.shape)
    
    # filter nan embeds
    print('filtering nan embeds')
    mask = ~np.isnan(embeds).any(axis=1)
    embeds = embeds[mask]
    keys = keys[mask]
    print('NaN filtered shapes', keys.size, embeds.shape)
    
    # get lcc
    print('getting lcc')
    g = lcc_excluding_no_content()
    
    # filter embeds to only include nodes in lcc
    print('filtering embeds')
    nodes = g.vs['_nx_name']
    mask = np.isin(keys, nodes)
    embeds = embeds[mask]
    keys = keys[mask]
    print('LCC filtered shapes', keys.size, embeds.shape)
    assert len(keys) == len(nodes)  # sanity check
    
    # community detection
    print('detecting communities')
    community = la.find_partition(g, la.ModularityVertexPartition, seed=0)
    num_communities = len(community)
    community_idx = np.argsort(nodes)
    original_idx = np.argsort(community_idx)
    community_labels = np.array(community.membership)[community_idx]
    print('Modularity', community.modularity)
    print('Silhouette', silhouette_score(embeds, community_labels))
    
    # clustering
    print('clustering')
    # clustering = AgglomerativeClustering(n_clusters=num_communities, linkage='single').fit(embeds)
    clustering = KMeans(n_clusters=num_communities, random_state=0).fit(embeds)
    clustering_labels = clustering.labels_
    print('Modularity', g.modularity(clustering_labels[original_idx]))
    print('Silhouette', silhouette_score(embeds, clustering_labels))
    
    # compare community and clustering
    print('comparing')
    nmi = normalized_mutual_info_score(community_labels, clustering_labels)
    ari = adjusted_rand_score(community_labels, clustering_labels)
    print(f'NMI: {nmi}')
    print(f'ARI: {ari}')

    # save results
    print('saving')
    data = {
        'nodes': np.sort(nodes),
        'community_labels': community_labels,
        'clustering_labels': clustering_labels
    }
    df = pd.DataFrame(data)
    df.to_csv('../data/clustering.csv', index=False)

    # tsne plot
    print('plotting')
    tsne = TSNE(n_components=2, random_state=0)
    embeds = tsne.fit_transform(embeds)
    
    fig, ax = plt.subplots()
    ax.scatter(embeds[:, 0], embeds[:, 1], c=clustering_labels, s=1, alpha=.1)
    ax.set_title('Clustering Labels')
    fig.savefig('../figs/clustering_scatter.png')
    
    fig, ax = plt.subplots()
    ax.scatter(embeds[:, 0], embeds[:, 1], c=community_labels, s=1, alpha=.1)
    ax.set_title('Community Labels')
    fig.savefig('../figs/community_scatter.png')
