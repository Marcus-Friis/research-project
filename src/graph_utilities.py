import igraph as ig


def base_graph():
    # load graph
    with open('../data/Cit-HepPh.txt', 'r') as f:
        for _ in range(4):
            f.readline()
        edges = []
        while True:
            line = f.readline()
            if not line:
                break
            edges.append(line.split())
    G = ig.Graph.TupleList(edges, directed=True)
    return G

def lcc():
    G = base_graph()
    G = G.components(mode='weak').giant()
    return G

def lcc_excluding_no_content():
    g = lcc()
    g.delete_vertices(g.vs.select(lambda x: x['name'] in ['9812218', '9305237']))
    return g

def aug_graph():
    with open('../data/edges.txt', 'r') as f:
        edges = []
        while True:
            line = f.readline()
            if not line:
                break
            edges.append(line.split())
        
    edges = [(edge[0], edge[1], edge[2].lower()) for edge in edges]

    g = ig.Graph.TupleList(edges, directed=True, edge_attrs=['label'])
    g = g.components(mode='weak').giant()
    g.delete_vertices(g.vs.select(lambda x: x['name'] in ['9812218', '9305237']))
    return g

def lcc_aug():
    g = aug_graph()
    g = g.components(mode='weak').giant()
    return g

def metropolis_hastings(seed=None):
    G = lcc_excluding_no_content()
    subgraph_nodes = metropolis_hastings_sampling(G, subgraph_size=G.vcount() // 10, seed=seed)
    G = G.subgraph(subgraph_nodes)
    return G

def forest_fire():
    G = lcc_excluding_no_content(seed=None)
    subgraph_nodes = forest_fire_sampling(G, subgraph_size=G.vcount() // 10)
    G = G.subgraph(subgraph_nodes)
    return G

def random_walk(seed=None):
    G = lcc_excluding_no_content(seed=None)
    subgraph_nodes = random_walk_sampling(G, subgraph_size=G.vcount() // 10, seed=seed)
    G = G.subgraph(subgraph_nodes)
    return G
