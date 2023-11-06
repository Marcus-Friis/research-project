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


def metropolis_hastings_sampling(g, start_node=None, subgraph_size=1000):
    # init starting node
    if start_node is None:
        current_node = random.choice(g.vs)
    else:
        current_node = start_node

    # simulate metropolis hastings random walk
    subgraph_nodes = [current_node.index]
    while len(np.unique(subgraph_nodes)) < subgraph_size:
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
        subgraph_nodes.append(current_node.index)

    return subgraph_nodes
