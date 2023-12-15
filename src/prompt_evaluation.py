import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pickle
import os

from sklearn.utils import resample
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters


if __name__ == '__main__':
    # load data paths
    path = '../data/prompt_engineering/'
    files = os.listdir(path)
    files = [f for f in files if 'explain' not in f]  # exclude explain file
    files = np.array(files)[[0, 4, 3, 1, 2]]  # custom order
    
    # load data
    data = {}
    for file in files:
        with open(path + file, 'rb') as f:
            file_data = pickle.load(f)
        entry = [[x.choices[0].message.content for x in n] for n in file_data]
        entry = [[x.lower().strip().replace('.', '') for x in n] for n in entry]  # clean entries
        data[file] = entry
        
    # get reproducible sample
    with open('../data/Cit-HepPh.txt', 'r') as f:
        for _ in range(4):
            next(f)
        edges = f.readlines()
    edges = [edge.strip().split('\t') for edge in edges]

    start = 0
    end = len(edges)
    size = 100
    np.random.seed(42)
    idx = np.random.uniform(start, end, size=size).astype(int)
    edges = np.array(edges)
    edges = edges[idx]


    # put data in dataframe
    df_data = {
        'source': edges[:, 0],
        'target': edges[:, 1]
    }
    for key in data.keys():
        df_data[key] = np.array(data[key]).T.tolist()
        
    df = pd.DataFrame(df_data)
    
    # calculate mode
    for file in files:
        df['mode_' + file] = df[file].apply(lambda x: max(set(x), key=x.count))
        
    print(df.head())
    
    # mode plot
    labels = ['\n'.join(file.split('_')).replace('.pkl', '') for file in files]

    fig, ax = plt.subplots()

    agree_counts = [(df['mode_'+file] == 'agreement').sum() for file in files]
    neutral_counts = [(df['mode_'+file] == 'neutral').sum() for file in files]
    disagree_counts = [(df['mode_'+file] == 'disagreement').sum() for file in files]
    # nota_counts = [((df['mode_'+file] != 'agreement') & (df['mode_'+file] != 'neutral') & (df['mode_'+file] != 'disagreement')).sum() for file in files]

    x_pos = np.arange(len(files))
    width = 0.2

    ax.bar(x_pos - width, agree_counts, width, label='agreement', color='tab:green')
    ax.bar(x_pos, neutral_counts, width, label='neutral', color='tab:blue')
    ax.bar(x_pos + width, disagree_counts, width, label='disagreement', color='tab:orange')
    # ax.bar(x_pos + 2*width, nota_counts, width, label='not applicable', color='tab:grey')

    ax.set_xticklabels(['If you read this, you are beautiful <3']+labels, rotation=45)  # the first part of the xticklabels is a hack to get the first label to show
    ax.set_title('Label distribution across\nprompt/model configurations')
    ax.set_xlabel('Prompt and Model')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    fig.savefig('../figs/model_mode_bar.svg')
    
    # calculate inter model agreement
    kappas = {}
    for key in data.keys():
        x = np.array(data[key]).T
        arr, categories = aggregate_raters(x)
        print(key, categories)
        kappas[key] = fleiss_kappa(arr)
        
    # bootstrap
    n_bootstraps = 1000
    bootstrapped_kappas = {}
    for key in data.keys():
        b_kappas = []
        for i in range(n_bootstraps):
            x = resample(np.array(data[key]).T)
            arr, categories = aggregate_raters(x)
            b_kappas.append(fleiss_kappa(arr))
        bootstrapped_kappas[key] = b_kappas
        
    # plot result
    fig, ax = plt.subplots()
    x, y = zip(*kappas.items())
    idx = np.argsort(y)
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    std = np.std(list(bootstrapped_kappas.values()), axis=1)
    std = std[idx]
    x = ['\n'.join(xx.replace('.pkl', '').split('_')) for xx in x]
    ax.barh(x, y, xerr=std, color='tab:blue')
    ax.set_title('Inter-prompt/model agreement\nFleiss Kappa for Prompt and Model')
    ax.set_xlabel('Fleiss Kappa')
    ax.set_ylabel('Prompt and Model')
    plt.tight_layout()
    fig.savefig('../figs/model_agreement.svg')
