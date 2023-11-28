import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from graph_utilities import lcc_excluding_no_content
import leidenalg as la

if __name__ == '__main__':
    # load embeds
    print('loading embeds')
    with open('../data/embeds.json', 'r') as f:
        embeds = json.load(f)
        
    keys = np.array(list(embeds.keys()))
    embeds = np.array(list(embeds.values()))
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
    
    # community detection
    print('detecting communities')
    community = la.find_partition(g, la.ModularityVertexPartition, seed=0)
    num_communities = len(community)
    community_idx = np.argsort(nodes)
    community_labels = np.array(community.membership)[community_idx]
    
    # clustering
    print('clustering')
    # clustering = AgglomerativeClustering(n_clusters=num_communities, linkage='single').fit(embeds)
    clustering = KMeans(n_clusters=num_communities, random_state=0).fit(embeds)
    clustering_idx = np.argsort(keys)
    clustering_labels = clustering.labels_[clustering_idx]
    
    # compare community and clustering
    print('comparing')
    nmi = normalized_mutual_info_score(community_labels, clustering_labels)
    ari = adjusted_rand_score(community_labels, clustering_labels)
    print(f'NMI: {nmi}')
    print(f'ARI: {ari}')
