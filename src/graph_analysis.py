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


ANALYSIS_LABEL = 'no_title_hep-ph'
PLOT_TITLES = ''  # 'lcc' + '\n'
PLOT_GRAPH = True
SEED = 42
G = base_graph()


if __name__ == '__main__':
    # make directory for figures if it doesn't exist
    if not os.path.exists(f'../figs/{ANALYSIS_LABEL}'):
        os.makedirs(f'../figs/{ANALYSIS_LABEL}')
        
    # COMMUNITY DETECTION
    communities = la.find_partition(G, la.ModularityVertexPartition, seed=SEED)

    # # GRAPH PROPERTIES
    # basic graph attributes
    with open(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-graph-properties.txt', 'w') as f:
        f.write(f'number of nodes\t {G.vcount()}\n')
        f.write(f'number of edges\t {G.ecount()}\n')
        f.write(f'average degree\t {np.mean(G.degree())}\n')
        f.write(f'median in-degree\t {np.median(G.indegree())}\n')
        f.write(f'median out-degree\t {np.median(G.outdegree())}\n')
        f.write(f'average path length\t {G.average_path_length(directed=False)}\n')
        f.write(f'diameter\t {G.diameter(directed=False)}\n')
        f.write(f'number of selfloops\t {sum(G.is_loop())}\n')
        f.write(f'number of weakly connected components\t {len(G.components(mode="weak"))}\n')
        f.write(f'number of strongly connected components\t {len(G.components(mode="strong"))}\n')
        f.write(f'average clustering coefficient\t {G.transitivity_avglocal_undirected(mode="zero")}\n')
        f.write(f'fraction of closed triads\t {G.transitivity_undirected()}\n')
        f.write(f'density of graph\t {G.density(loops=False)}\n')
        f.write(f'assortativity coefficient\t {G.assortativity_degree(directed=False)}\n')
        f.write(f'has multiple edges\t {G.simplify(multiple=False).has_multiple()}\n')
        f.write(f'is dag\t {G.is_dag()}\n')
        f.write(f'modularity of graph\t {communities.modularity}\n')
        
    # DEGREE DISTRIBUTION
    # get list of in- and out-degrees
    in_degrees = G.indegree()
    out_degrees = G.outdegree()

    # get frequency of each degree and normalize to density
    d_in, v_in = np.unique(in_degrees, return_counts=True)
    v_in = v_in / v_in.sum()
    d_out, v_out = np.unique(out_degrees, return_counts=True)
    v_out = v_out / v_out.sum()

    x_upper_bound = max(np.max(d_in), np.max(d_out))
    y_upper_bound = max(np.max(v_in), np.max(v_out))

    dot_size = 15

    # plot in-degree
    fig, ax = plt.subplots(figsize=[6.4*0.75, 4.8*0.75])
    ax.scatter(d_in, v_in, s=dot_size)
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    # ax.set_title(f'{PLOT_TITLES}in-degree distribution')
    ax.set_xlim(-10, x_upper_bound + 10)
    ax.set_ylim(-0.01, y_upper_bound + 0.01)
    plt.tight_layout()
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-in-degree.svg')

    # plot out-degree
    fig, ax = plt.subplots(figsize=[6.4*0.75, 4.8*0.75])
    ax.scatter(d_out, v_out, s=dot_size)
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    # ax.set_title(f'{PLOT_TITLES}out-degree distribution')
    ax.set_xlim(-10, x_upper_bound + 10)
    ax.set_ylim(-0.01, y_upper_bound + 0.01)
    plt.tight_layout()
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-out-degree.svg')

    # plot in-degree log-log
    fig, ax = plt.subplots(figsize=[6.4*0.75, 4.8*0.75])
    ax.scatter(d_in, v_in, s=dot_size)
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    # ax.set_title(f'{PLOT_TITLES}in-degree distribution')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, x_upper_bound*2)
    ax.set_ylim(0.0001, y_upper_bound*2)
    plt.tight_layout()
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-log-in-degree.svg')

    # plot out-degree log-log
    fig, ax = plt.subplots(figsize=[6.4*0.75, 4.8*0.75])
    d_out, v_out = np.unique(out_degrees, return_counts=True)
    v_out = v_out / v_out.sum()
    ax.scatter(d_out, v_out, s=dot_size)
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    # ax.set_title(f'{PLOT_TITLES}out-degree distribution')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, x_upper_bound*2)
    ax.set_ylim(0.0001, y_upper_bound*2)
    plt.tight_layout()
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-log-out-degree.svg')

    # get complementray cumulative sum of degrees
    v_in_cum = 1 - np.cumsum(v_in)
    v_out_cum = 1 - np.cumsum(v_out)
    y_axis_lower_bound = min(np.min(v_in_cum), np.min(v_out_cum))

    # plot in-degree CCDF log-log
    fig, ax = plt.subplots(figsize=[6.4*0.75, 4.8*0.75])
    ax.plot(d_in, v_in_cum)
    # ax.set_title(f'{PLOT_TITLES}in-degree CCDF')
    ax.set_xlabel('k')
    ax.set_ylabel('p(k$\geq$x)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, x_upper_bound*2)
    ax.set_ylim(y_axis_lower_bound*2, 1.1)
    plt.tight_layout()
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-in-degree-ccdf.svg')

    # plot out-degree CCDF log-log
    fig, ax = plt.subplots(figsize=[6.4*0.75, 4.8*0.75])
    ax.plot(d_out, v_out_cum)
    # ax.set_title(f'{PLOT_TITLES}out-degree CCDF')
    ax.set_xlabel('k')
    ax.set_ylabel('p(k$\geq$x)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, x_upper_bound*2)
    ax.set_ylim(y_axis_lower_bound*2, 1.1)
    plt.tight_layout()
    fig.savefig(f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-out-degree-ccdf.svg')

    # visualize graph
    if PLOT_GRAPH:
        target = f'../figs/{ANALYSIS_LABEL}/{ANALYSIS_LABEL}-graph.svg'
        G = G.as_undirected()
        layout = G.layout("fr")
        ig.plot(communities, layout=layout, vertex_size=2, vertex_label=None, vertex_frame_width=0, 
                edge_arrow_size=0, edge_width=0.02, target=target)
