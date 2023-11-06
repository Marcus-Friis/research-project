import networkx as nx
import igraph as ig
import numpy as np
import random


def random_walk_sampling(g, teleportation_rate=0.05, start_node=None, subgraph_size=1000):
    # init starting node
    if start_node is None:
        current_node = random.choice(g.vs)
    else:
        current_node = start_node

    # simulate random walk
    subgraph_nodes = [current_node.index]
    while len(np.unique(subgraph_nodes)) < subgraph_size:
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

        subgraph_nodes.append(current_node.index)

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
