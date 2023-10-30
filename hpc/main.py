from llama_cpp import Llama
import json
import time
import pandas as pd

# prepare model
model_path = 'openbuddy-llama2-13b-v11.1.Q5_K_M.gguf'
n_gpu_layers = -1
embedding = True
verbose = False
n_ctx = 1024
n_batch = 1024
llama = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, embedding=embedding, verbose=verbose, n_ctx=n_ctx, n_batch=n_batch)

# open arxiv data
df = pd.read_csv('../data/arxiv.csv')
if 'embedding' not in df.columns:
    df['embedding'] = None

# generate embeddings
a = time.time()
try:
    for i, row in df.iterrows():
        if pd.isna(row.embedding):
            print('ITEM\t', i)
            abstract = row.abstract
            prompt = f'{abstract}'
            embedding = llama.embed(prompt)
            df.at[i, 'embedding'] = embedding
            df.to_csv('../data/arxiv2.csv')
except:
    df.to_csv('../data/arxiv2.csv')

b = time.time()

print('EXECUTED IN', b-a)
