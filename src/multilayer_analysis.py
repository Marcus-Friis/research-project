import igraph as ig
import leidenalg as la
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from graph_utilities import multilayer_lcc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score


if __name__ == '__main__':
    labels = ['agreement', 'neutral', 'disagreement']
    g, g_pos, g_neu, g_neg = multilayer_lcc(load_embeds=False)
    
    # # compute average distance between nodes of the same label
    # for label in labels:
    #     print(label)
    #     distances = []
    #     es = g.es.select(label_eq=label)
    #     for edge in es:
    #         u = g.vs[edge.source]
    #         v = g.vs[edge.target]
            
    #         emb_u = u['embedding'].reshape(-1, 1)
    #         emb_v = v['embedding'].reshape(-1, 1)
            
    #         similarity = cosine_similarity(emb_u, emb_v)
    #         distances.append(similarity)
    #     print(np.mean(distances))


    def get_degree_sequence(label):
        _, in_degrees = np.unique([edge.target for edge in g.es.select(label_eq=label)], return_counts=True)
        _, out_degrees = np.unique([edge.source for edge in g.es.select(label_eq=label)], return_counts=True)
        
        d_in, v_in = np.unique(in_degrees, return_counts=True)
        d_out, v_out = np.unique(out_degrees, return_counts=True)
        
        return d_in, v_in, d_out, v_out

    a_d_in, a_v_in, a_d_out, a_v_out = get_degree_sequence('agreement')
    n_d_in, n_v_in, n_d_out, n_v_out = get_degree_sequence('neutral')
    d_d_in, d_v_in, d_d_out, d_v_out = get_degree_sequence('disagreement')

    # normalize
    in_total = np.sum(a_v_in) + np.sum(n_v_in) + np.sum(d_v_in)
    out_total = np.sum(a_v_out) + np.sum(n_v_out) + np.sum(d_v_out)
    a_v_in = a_v_in / in_total
    n_v_in = n_v_in / in_total
    d_v_in = d_v_in / in_total
    a_v_out = a_v_out / out_total
    n_v_out = n_v_out / out_total
    d_v_out = d_v_out / out_total
    
    dot_size = 15

    # plot in-degree
    fig, ax = plt.subplots(figsize=[6.4*0.75, 4.8*0.75])
    ax.scatter(a_d_in, a_v_in, s=dot_size, color='tab:green', label='agreement', alpha=0.7, marker='^')
    ax.scatter(n_d_in, n_v_in, s=dot_size, color='tab:blue', label='neutral', alpha=0.7, marker='s')
    ax.scatter(d_d_in, d_v_in, s=dot_size, color='tab:orange', label='disagreement', alpha=0.7, marker='o')
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    # ax.set_title(f'LCC multilayer\nin-degree distribution')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, max(np.max(a_d_in), np.max(n_d_in), np.max(d_d_in), np.max(a_d_out), np.max(n_d_out), np.max(d_d_out))*2)
    ax.set_ylim(0.00001, max(np.max(a_v_in), np.max(n_v_in), np.max(d_v_in), np.max(a_v_out), np.max(n_v_out), np.max(d_v_out))*2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'../figs/mlg-in-degree.svg')

    # plot out-degree
    fig, ax = plt.subplots(figsize=[6.4*0.75, 4.8*0.75])
    ax.scatter(a_d_out, a_v_out, s=dot_size, color='tab:green', label='agreement', alpha=0.7, marker='^')
    ax.scatter(n_d_out, n_v_out, s=dot_size, color='tab:blue', label='neutral', alpha=0.7, marker='s')
    ax.scatter(d_d_out, d_v_out, s=dot_size, color='tab:orange', label='disagreement', alpha=0.7, marker='o')
    ax.set_xlabel('k')
    ax.set_ylabel('p(k)')
    # ax.set_title(f'LCC multilayer\nout-degree distribution')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, max(np.max(a_d_in), np.max(n_d_in), np.max(d_d_in), np.max(a_d_out), np.max(n_d_out), np.max(d_d_out))*2)
    ax.set_ylim(0.00001, max(np.max(a_v_in), np.max(n_v_in), np.max(d_v_in), np.max(a_v_out), np.max(n_v_out), np.max(d_v_out))*2)
    plt.tight_layout()
    fig.savefig(f'../figs/mlg-out-degree.svg')
    
    membership, improv = la.find_partition_multiplex(
        [g_pos, g_neu, g_neg],
        la.ModularityVertexPartition,
        layer_weights=[1, .5, -1],
        seed=42)
    
    print('Full graph\t', g.modularity(membership))
    print('Positive  \t', g_pos.modularity(membership))
    print('Neutral   \t', g_neu.modularity(membership))
    print('Negative  \t', g_neg.modularity(membership))
    
    print('Silhouette score\t', silhouette_score(g.vs['embedding'], membership, metric='cosine'))
