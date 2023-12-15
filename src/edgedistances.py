from graph_utilities import lcc_aug_embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    from collections import defaultdict
    data = defaultdict(int)
    
    g = lcc_aug_embedding(load_embeds=False)
    for edge in g.es:
        label = edge['label']
        u = g.vs[edge.source]
        v = g.vs[edge.target]
        u_emb = np.array(u['embedding']).reshape(1, -1)
        v_emb = np.array(v['embedding']).reshape(1, -1)
        
        data[label + '_euclidean'] += np.linalg.norm(u_emb - v_emb)
        data[label + '_cosine'] += cosine_similarity(u_emb, v_emb)[0,0]
        data[label + '_count'] += 1
        
    labels = ('agreement', 'neutral', 'disagreement')
    for label in labels:
        print(label)
        print('\tEuclidean\t', data[label + '_euclidean'] / data[label + '_count'])
        print('\tCosine   \t', data[label + '_cosine'] / data[label + '_count'])