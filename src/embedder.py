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
with open('embeds.json', 'r') as f:
    embeds = json.load(f)

# generate embeddings
a = time.time()
for k, (i, row) in enumerate(df.iterrows()):
    article_id = str(row.id)
    if article_id not in embeds:
        print(k, '\tITEM\t', i, article_id)
        abstract = row.abstract
        prompt = f'{abstract}'
        embedding = llama.embed(prompt)
        embeds[article_id] = embedding

        # dump every 500th
        if k % 500 == 0:
            print('DUMPING')
            with open('embeds.json', 'w') as f:
                json.dump(embeds, f, indent=4)

b = time.time()

print('EXECUTED IN', b-a)
