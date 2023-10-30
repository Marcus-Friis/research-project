import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the JSON file
json_file = "lol.json"

try:
    with open(json_file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("File %s not found. Please make sure the file exists." % json_file)
    exit(1)

# Extract embeddings
keys = list(data.keys())
embeddings = [data[key]['embedding'] for key in keys]

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(embeddings)

# Create a DataFrame for plotting
df = pd.DataFrame({'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1], 'key': keys})

# Plot the scatterplot
plt.figure(figsize=(10, 8))
plt.scatter(df['x'], df['y'], alpha=0.5)

# Annotate points with their keys
for i, row in df.iterrows():
    plt.annotate(row['key'], (row['x'], row['y']), fontsize=8)

plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
# Save the plot as a PNG file
plt.savefig("tsne_plot.png")

