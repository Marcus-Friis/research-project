import networkx as nx
import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import random
from collections import deque

import os
import sys
sys.path.append('..')
from src.graph_utilities import *

ANALYSIS_LABEL = 'lcc'
PLOT_TITLES = 'lcc'
PLOT_GRAPH = True

def base_graph():
    # load graph
    G_nx = nx.read_edgelist('../data/Cit-HepPh.txt', create_using=nx.DiGraph)
    G = ig.Graph.from_networkx(G_nx)
    return G

def lcc():
    G = base_graph()
    G = G.components(mode='weak').giant()
    return G

def metropolis_hastings():
    G = lcc()
    subgraph_nodes = metropolis_hastings_sampling(G, subgraph_size=G.vcount() / 10)
    G = G.subgraph(subgraph_nodes)
    return G

if __name__ == '__main__':
    # CHANGE THIS LINE TO CHANGE GRAPH
    G = lcc()

    # make directory for figures if it doesn't exist
    if not os.path.exists(f'../figs/{ANALYSIS_LABEL}'):
        os.makedirs(f'../figs/{ANALYSIS_LABEL}')

    # # GRAPH PROPERTIES
    # # basic graph attributes
    print('number of nodes', G.vcount())
    print('number of edges', G.ecount())
    print('number of selfloops', sum(G.is_loop()))
    print('number of weakly connected components', len(G.components(mode='weak')))
    print('number of strongly connected components', len(G.components(mode='strong')))
    print('average clustering coefficient', G.transitivity_avglocal_undirected())
    print('fraction of closed triads', G.transitivity_undirected())
    print('density of graph', G.density(loops=False))


    # DEGREE DISTRIBUTION
    # get list of in- and out-degrees
    in_degrees = G.indegree()
    out_degrees = G.outdegree()

    # print basic degree statistics
    # print("Highest indegree: ", max(in_degrees))
    # print("Highest outdegree: ", max(out_degrees),"\n")

    # print("Lowest indegree: ", min(in_degrees))
    # print("Lowest outdegree: ", min(out_degrees),"\n")

    # print(f"Average in-degree: {np.mean(in_degrees)}")
    # print(f"Average out-degree: {np.mean(out_degrees)}")

    # get frequency of each degree and normalize to density
    d_in, v_in = np.unique(in_degrees, return_counts=True)
    v_in = v_in / v_in.sum()
    d_out, v_out = np.unique(out_degrees, return_counts=True)
    v_out = v_out / v_out.sum()

    x_upper_bound = max(np.max(d_in), np.max(d_out))
    y_upper_bound = max(np.max(v_in), np.max(v_out))

    dot_size = 15

    # plot in-degree
    fig, ax = plt.subplots()
    ax.scatter(d_in, v_in, s=dot_size)
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    ax.set_title(f'{PLOT_TITLES}\nin-degree distribution')
    ax.set_xlim(-10, x_upper_bound + 10)
    ax.set_ylim(-0.01, y_upper_bound + 0.01)
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-in-degree.svg')

    # plot out-degree
    fig, ax = plt.subplots()
    ax.scatter(d_out, v_out, s=dot_size)
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    ax.set_title(f'{PLOT_TITLES}\nout-degree distribution')
    ax.set_xlim(-10, x_upper_bound + 10)
    ax.set_ylim(-0.01, y_upper_bound + 0.01)
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-out-degree.svg')

    # plot in-degree log-log
    fig, ax = plt.subplots()
    ax.scatter(d_in, v_in, s=dot_size)
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    ax.set_title(f'{PLOT_TITLES}\nin-degree distribution')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, x_upper_bound*2)
    ax.set_ylim(0.0001, y_upper_bound*2)
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-log-in-degree.svg')

    # plot out-degree log-log
    fig, ax = plt.subplots()
    d_out, v_out = np.unique(out_degrees, return_counts=True)
    v_out = v_out / v_out.sum()
    ax.scatter(d_out, v_out, s=dot_size)
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    ax.set_title(f'{PLOT_TITLES}\nout-degree distribution')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, x_upper_bound*2)
    ax.set_ylim(0.0001, y_upper_bound*2)
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-log-out-degree.svg')

    # get complementray cumulative sum of degrees
    v_in_cum = 1 - np.cumsum(v_in)
    v_out_cum = 1 - np.cumsum(v_out)
    y_axis_lower_bound = min(np.min(v_in_cum), np.min(v_out_cum))

    # plot in-degree CCDF log-log
    fig, ax = plt.subplots()
    ax.plot(d_in, v_in_cum)
    ax.set_title(f'{PLOT_TITLES}\nin-degree CCDF')
    ax.set_xlabel('k')
    ax.set_ylabel('p(k$\geq$x)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, x_upper_bound*2)
    ax.set_ylim(y_axis_lower_bound*2, 1.1)
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-in-degree-ccdf.svg')

    # plot out-degree CCDF log-log
    fig, ax = plt.subplots()
    ax.plot(d_out, v_out_cum)
    ax.set_title(f'{PLOT_TITLES}\nout-degree CCDF')
    ax.set_xlabel('k')
    ax.set_ylabel('p(k$\geq$x)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, x_upper_bound*2)
    ax.set_ylim(y_axis_lower_bound*2, 1.1)
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-out-degree-ccdf.svg')

    # COMMUNITY DETECTION
    communities = la.find_partition(G, la.ModularityVertexPartition)
    print('Modularity', G.modularity(communities))

    # visualize graph
    if PLOT_GRAPH:
        target = f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-graph.svg'
        layout = G.layout("fr")
        ig.plot(communities, layout=layout, vertex_size=0, vertex_label=None, vertex_frame_width=0, 
                edge_arrow_size=0.02, edge_width=0.02, target=target)
    