import networkx as nx
import igraph as ig
import numpy as np
import random


def random_walk_sampling(g, teleportation_rate=0.05, start_node=None, subgraph_size=1000):
    if start_node is None:
        # init starting node
        start_node = random.choice(g.vs)
    
    subgraph_nodes = []
    while len(np.unique(subgraph_nodes)) < subgraph_size:
        # simulate path length
        steps = 0
        while random.random() > teleportation_rate:
            steps += 1
        # do random walk
        walk_nodes = g.random_walk(start_node, steps=steps)
        subgraph_nodes += walk_nodes
        start_node = random.choice(g.vs)
    
    return subgraph_nodes
