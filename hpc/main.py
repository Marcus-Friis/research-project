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

# embeds-json
try:
    with open('embeds.json', 'r') as f:
        embeds = json.load(f)
except FileNotFoundError:
    print('NO FILE FOUND, CREATING NEW')
    with open('embeds.json', 'w') as f:
        embeds = {}
        json.dump(embeds, f, indent=4)

# generate embeddings
a = time.time()
try:
    for i, row in df.iterrows():
        if row.id not in embeds:
            print('ITEM\t', i)
            article_id = row.id
            abstract = row.abstract
            prompt = f'{abstract}'
            embedding = llama.embed(prompt)
            embeds[article_id] = embedding
            with open('embeds.json', 'w') as f:
                json.dump(embeds, f, indent=4)
except Exception as err:
    print('FUCK ERROR', err)
    with open('embeds.json', 'w') as f:
        json.dump(embeds, f, indent=4)

b = time.time()

print('EXECUTED IN', b-a)
