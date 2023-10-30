import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

json_file = "lol.json"
with open(json_file, 'r') as f:
    data = json.load(f)

keys = list(data.keys())
print('NUM ARTICLES', len(keys))
embeddings = [data[key]['embedding'] for key in keys]

kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)
clusters = kmeans.labels_

reducer = TSNE(n_components=2, random_state=0)
embeddings = reducer.fit_transform(embeddings)

fig, ax = plt.subplots()
ax.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters)
plt.show()
