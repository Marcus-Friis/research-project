import igraph as ig
import numpy as np
import random
from collections import deque


def random_walk_sampling(g, teleportation_rate=0.05, start_node=None, subgraph_size=1000, seed=None):
    if seed is not None:
        random.seed(seed)
        
    # init starting node
    if start_node is None:
        current_node = random.choice(g.vs)
    else:
        current_node = start_node

    # simulate random walk
    subgraph_nodes = set()
    subgraph_nodes.add(current_node.index)
    while len(subgraph_nodes) < subgraph_size:
        # get neighbors of current node
        neighbors = current_node.neighbors()
        num_neighbors = len(neighbors)

        # decide whether to teleport or not
        if random.random() < teleportation_rate:
            current_node = random.choice(g.vs)
        else:
            # pick a neighbor
            picked_neighbor = random.choice(neighbors)
            current_node = picked_neighbor

        subgraph_nodes.add(current_node.index)

    return subgraph_nodes


def metropolis_hastings_sampling(g, start_node=None, subgraph_size=1000, seed=None):
    if seed is not None:
        random.seed(seed)
        
    # init starting node
    if start_node is None:
        current_node = random.choice(g.vs)
    else:
        current_node = start_node

    # simulate metropolis hastings random walk
    subgraph_nodes = set()
    subgraph_nodes.add(current_node.index)
    while len(subgraph_nodes) < subgraph_size:
        # get neighbors of current node
        neighbors = current_node.neighbors()
        num_neighbors = len(neighbors)

        # pick a neighbor
        while True:
            picked_neighbor = random.choice(neighbors)
            neighbors_neighbors = picked_neighbor.neighbors()
            num_neighbors_neighbors = len(neighbors_neighbors)
            # calculate probability of walking to picked neighbor
            prob = num_neighbors / num_neighbors_neighbors
            if random.random() <= prob:
                break
        current_node = picked_neighbor
        subgraph_nodes.add(current_node.index)

    return subgraph_nodes


'''
Below code is for forest fire sampling
'''
def geometric_random_number(p_forward):
    return np.random.geometric(p_forward)

def forest_fire_sampling(G, subgraph_size: int, p_forward: float = 0.1):
    sampled_nodes = set()

    # Function to start/restart the fire with a random node
    def start_fire():
        # Start with a random node not already sampled
        remaining_nodes = [v.index for v in G.vs if v.index not in sampled_nodes]
        return random.choice(remaining_nodes)

    # Queue for nodes to visit
    to_visit = deque([start_fire()])
    
    while len(sampled_nodes) < subgraph_size:
        if not to_visit:
            to_visit.append(start_fire())  # Restart the fire if needed
        
        current_node = to_visit.popleft()
        if current_node in sampled_nodes:
            continue
        
        # Add the current node to the set of sampled nodes
        sampled_nodes.add(current_node)

        # Get the neighbors for an undirected graph
        neighbors = G.neighbors(current_node)
        
        # Draw a number from the geometric distribution for the number of edges to follow
        x = geometric_random_number(p_forward)
        nW = min(x, len(neighbors))
        
        # Select nW neighbors to follow
        selected_neighbors = random.sample(neighbors, nW) if nW < len(neighbors) else neighbors
        for neighbor in selected_neighbors:
            if len(sampled_nodes) >= subgraph_size:
                break
            if neighbor not in sampled_nodes:
                to_visit.append(neighbor)
    
   
    return sampled_nodes
'''
ends here
'''

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
