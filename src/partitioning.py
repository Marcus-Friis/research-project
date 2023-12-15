import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
from graph_utilities import multilayer_lcc
import leidenalg as la

if __name__ == '__main__':
    # go away warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # cli stuff
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--no-plot', action='store_false', help='Plot plots')
    parser.add_argument('-l', '--no-embeds', action='store_false', help='Load embeddings or use random')
    parser.add_argument('-d', '--no-dump', action='store_false', help='Dont dump to file')
    args = parser.parse_args()
    PLOT = args.no_plot
    LOAD_EMBEDS = args.no_embeds
    DUMP = args.no_dump
    
    # seed for reproducibility
    seed = 42
    
    # load graph
    print('loading graph')
    g, g_pos, g_neu, g_neg = multilayer_lcc(load_embeds=LOAD_EMBEDS)
    nodes = g.vs['name']
    embeds = np.array(g.vs['embedding'])
    
    # community detection
    print('detecting communities')
    community = la.find_partition(g, la.ModularityVertexPartition, seed=seed)
    num_communities = len(community)
    community_labels = community.membership
    print('\tModularity', community.modularity)
    print('\t\tAgreement', g_pos.modularity(community_labels))
    print('\t\tNeutral', g_neu.modularity(community_labels))
    print('\t\tDisagreement', g_neg.modularity(community_labels))
    print('\tSilhouette', silhouette_score(embeds, community_labels))
    
    # multi-layer community detection
    print('detecting multi-layer communities')
    ml_community_labels, improv = la.find_partition_multiplex(
        [g_pos, g_neu, g_neg],
        la.ModularityVertexPartition,
        layer_weights=[1, .5, -1],
        seed=seed)
    print('\tImprovement', improv)
    print('\tModularity', g.modularity(ml_community_labels))
    print('\t\tAgreement', g_pos.modularity(ml_community_labels))
    print('\t\tNeutral', g_neu.modularity(ml_community_labels))
    print('\t\tDisagreement', g_neg.modularity(ml_community_labels))
    print('\tSilhouette', silhouette_score(embeds, ml_community_labels))
    
    # compare community and multi-layer community
    print('comparing to multi-layer community')
    nmi = normalized_mutual_info_score(community_labels, ml_community_labels)
    ami = adjusted_mutual_info_score(community_labels, ml_community_labels)
    ari = adjusted_rand_score(community_labels, ml_community_labels)
    print(f'\tNMI: {nmi}')
    print(f'\tAMI: {ami}')
    print(f'\tARI: {ari}')
    
    # dimensionality reduction
    if PLOT:
        print('reducing dimensionality')
        tsne = TSNE(n_components=2, random_state=seed)
        tsne_embeds = tsne.fit_transform(embeds)
    
    # community plot
    if PLOT:
        print('plotting community')
        fig, ax = plt.subplots()
        ax.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], c=community_labels, s=1, alpha=.1)
        ax.set_title('Community Labels')
        fig.savefig('../figs/community_scatter.png')
        
        print('plotting multi-layer community')
        fig, ax = plt.subplots()
        ax.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], c=ml_community_labels, s=1, alpha=.1)
    
    data = {
        'nodes': nodes,
        'community_labels': community_labels,
        'ml_community_labels': ml_community_labels,
    }
    
    for k in [5, 10, 20, 50, 100]:
        # clustering
        print(f'clustering k={k}')
        clustering = KMeans(n_clusters=k, random_state=seed).fit(embeds)
        clustering_labels = clustering.labels_
        data[f'\tclustering_labels_{k}'] = clustering_labels
        print('\tModularity', g.modularity(clustering_labels))
        print('\t\tAgreement', g_pos.modularity(clustering_labels))
        print('\t\tNeutral', g_neu.modularity(clustering_labels))
        print('\t\tDisagreement', g_neg.modularity(clustering_labels))
        print('\tSilhouette', silhouette_score(embeds, clustering_labels))
        
        # compare community and clustering
        print('\tcomparing to community')
        nmi = normalized_mutual_info_score(community_labels, clustering_labels)
        ami = adjusted_mutual_info_score(community_labels, clustering_labels)
        ari = adjusted_rand_score(community_labels, clustering_labels)
        print(f'\t\tNMI: {nmi}')
        print(f'\t\tAMI: {ami}')
        print(f'\t\tARI: {ari}')
        
        # compare multi-layer community and clustering
        print('\tcomparing to multi-layer community')
        nmi = normalized_mutual_info_score(ml_community_labels, clustering_labels)
        ami = adjusted_mutual_info_score(ml_community_labels, clustering_labels)
        ari = adjusted_rand_score(ml_community_labels, clustering_labels)
        print(f'\t\tNMI: {nmi}')
        print(f'\t\tAMI: {ami}')
        print(f'\t\tARI: {ari}')

        # plot clustering
        if PLOT:
            print('\tplotting clustering')
            fig, ax = plt.subplots()
            ax.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], c=clustering_labels, s=1, alpha=.1)
            ax.set_title(f'T-SNE Clustering, $k$={k}')
            fig.savefig(f'../figs/clustering_{k}_scatter.png')
    
    # save results
    if DUMP:
        print('saving')
        df = pd.DataFrame(data)
        df.to_csv('../data/partitions.csv', index=False)
