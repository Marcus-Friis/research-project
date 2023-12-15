import os
import numpy as np

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--plot', default=False, help='Plot plots')
    parser.add_argument('-d', '--dump', default=True, help='Dump to file')
    args = parser.parse_args()
    PLOT = bool(args.plot)
    DUMP = bool(args.dump)
    
    # get all edge files
    path = '../data/edges/'
    files = [file for file in os.listdir(path) if 'edges_' in file]

    # read all edges
    edges = []
    for file in files:
        with open(path + file, 'r') as f:
            file_edges = f.readlines()
        
        # elementary cleaning
        file_edges = [edge.strip().split('\t') for edge in file_edges]
        edges += file_edges
        
    edges = np.array(edges)
    
    # elementary cleaning part 2
    edges_clean = np.char.strip(np.char.lower(edges[:, 2]))
    
    # get masks for each label
    disagree_mask = np.char.find(edges_clean, 'disagree') != -1
    neutral_mask = np.char.find(edges_clean, 'neutral') != -1
    agree_mask = (np.char.find(edges_clean, 'agree') != -1) & (~disagree_mask) & (~neutral_mask)
    none_mask = np.char.find(edges_clean, 'none') != -1
    
    # assign correct labels
    edges_clean[disagree_mask] = 'disagreement'
    edges_clean[neutral_mask] = 'neutral'
    edges_clean[agree_mask] = 'agreement'
    edges_clean[(~disagree_mask) & (~neutral_mask) & (~agree_mask) & (~none_mask)] = 'neutral'
    
    # safety checks
    agree_check = edges_clean == 'agreement'
    neutral_check = edges_clean == 'neutral'
    disagree_check = edges_clean == 'disagreement'
    none_check = edges_clean == 'none'
    
    assert np.all(disagree_check | neutral_check | agree_check | none_check)
    assert (disagree_check & neutral_check).sum() == 0
    assert (disagree_check & agree_check).sum() == 0
    assert (neutral_check & agree_check).sum() == 0
    assert (disagree_check & none_check).sum() == 0
    assert (neutral_check & none_check).sum() == 0
    assert (agree_check & none_check).sum() == 0
    
    if DUMP:
        # dump to file
        edges[:, 2] = edges_clean
        np.savetxt('../data/edges.txt', edges, fmt='%s')

    if PLOT:
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        
        # plot label distribution
        x, y = np.unique(edges_clean, return_counts=True)
        idx = np.argsort(y)[::-1]
        x = x[idx]
        y = y[idx]

        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.set_title('Cit-HEP-PH-Aug\nEdge label distribution')
        ax.set_ylabel('Count')
        ax.set_xlabel('Label')
        for i, v in enumerate(y):
            ax.text(i, v+1000, str(v), fontweight='semibold', ha='center')
        
        plt.tight_layout()
        fig.savefig('../figs/label_distribution.svg')
