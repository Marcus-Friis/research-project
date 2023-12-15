from graph_utilities import multilayer_lcc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    g, g_pos, g_neu, g_neg = multilayer_lcc()
    
    # compute average distance between nodes of the same label
    labels = ('agreement', 'neutral', 'disagreement')
    for label in labels:
        print(label)
        distances = []
        similarities = []
        es = g.es.select(label_eq=label)
        for edge in es:
            u = g.vs[edge.source]
            v = g.vs[edge.target]
            u_emb = np.array(u['embedding']).reshape(-1, 1)
            v_emb = np.array(v['embedding']).reshape(-1, 1)
            distances.append(np.linalg.norm(u_emb - v_emb))
            similarities.append(cosine_similarity(u_emb, v_emb))
        print('\tEuclidean\t', np.mean(distances))
        print('\tCosine   \t', np.mean(similarities))
