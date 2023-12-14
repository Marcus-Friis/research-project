import igraph as ig
import leidenalg as la
from graph_utilities import *


def community_summary(community):
    print(f'number of communities: {len(community)}')
    print(f'largest community size: {max(community.sizes())}')
    print(f'modularity: {community.modularity}')

if __name__ == '__main__':
    g = lcc_excluding_no_content()
    g_un = g.as_undirected()
    
    community = g_un.community_multilevel()
    print('louvain')
    community_summary(community)
    
    community = la.find_partition(g_un, la.ModularityVertexPartition, seed=42)
    print('leiden')
    community_summary(community)
    
    community = g_un.community_infomap()
    print('infomap')
    community_summary(community)

    community = g_un.community_fastgreedy().as_clustering()
    print('fastgreedy')
    community_summary(community)
    
    community = g_un.community_label_propagation()
    print('label propagation')
    community_summary(community)

    community = g_un.community_walktrap().as_clustering()
    print('community_walktrap')
    community_summary(community)
