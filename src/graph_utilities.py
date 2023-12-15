import igraph as ig
import numpy as np
import json


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
    with open('../data/Cit-HEP-PH-Aug.txt', 'r') as f:
        edges = []
        while True:
            line = f.readline()
            if not line:
                break
            edges.append(line.split())
        
    edges = [(edge[0], edge[1], edge[2].lower()) for edge in edges]

    g = ig.Graph.TupleList(edges, directed=True, edge_attrs=['label'])
    g.delete_vertices(g.vs.select(lambda x: x['name'] in ['9812218', '9305237']))
    return g

def lcc_aug():
    g = aug_graph()
    g = g.components(mode='weak').giant()
    return g

def lcc_aug_embedding(load_embeds=True):
    g = lcc_aug()
    
    if load_embeds:
        with open('../data/embeds.json', 'r') as f:
            embeddings = json.load(f)
        g.vs['embedding'] = [embeddings[str(node)] for node in g.vs['name']]
    else:
        g.vs['embedding'] = np.random.rand(g.vcount(), 10)
    return g

def multilayer_lcc(load_embeds=True):
    g = lcc_aug_embedding(load_embeds=load_embeds)
    g_pos = g.subgraph_edges(g.es.select(label_eq='agreement'), delete_vertices=False)
    g_neu = g.subgraph_edges(g.es.select(label_eq='neutral'), delete_vertices=False)
    g_neg = g.subgraph_edges(g.es.select(label_eq='disagreement'), delete_vertices=False)
    return g, g_pos, g_neu, g_neg