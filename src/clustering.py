import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from graph_utilities import lcc_excluding_no_content

if __name__ == '__main__':
    with open('../data/embeds.json', 'r') as f:
        embeds = json.load(f)
        
    keys = list(embeds.keys())
    embeds = np.array(embeds.values())
    
    print(np.isnan(embeds).any())
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage='single').fit(embeds)
    
    print(clustering.labels_)
    
    # g = lcc_excluding_no_content()
    # ids = g.vs['_nx_name']

    
