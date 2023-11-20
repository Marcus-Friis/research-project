import igraph as ig
import leidenalg as la
from graph_utilities import *


def community_summary(community):
    print(f'number of communities: {len(community)}')
    print(f'largest community size: {max(community.sizes())}')
    print(f'modularity: {community.modularity}')

if __name__ == '__main__':
    g = lcc()
    g_un = g.as_undirected()
    
    community = la.find_partition(g, la.ModularityVertexPartition)
    print('leiden')
    community_summary(community)
    
    community = g.community_infomap()
    print('infomap')
    community_summary(community)

    community = g_un.community_fastgreedy().as_clustering()
    print('fastgreedy')
    community_summary(community)
    