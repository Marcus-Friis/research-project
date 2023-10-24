from llama_cpp import Llama
import json

# prepare model
model_path = 'openbuddy-llama2-13b-v11.1.Q5_K_M.gguf'
n_gpu_layers = -1
embedding = True
verbose = False
llama = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, embedding=embedding, verbose=verbose)

# # define prompt
# prompt = 'Q: who is the most beautiful? Mads or Morten? A:'

# # generate text
# output = llama(prompt)
# print(output)

# # generate embedding
# embedding = llama.embed(prompt)
# print(embedding)

# open arxiv data
with open('../data/arxiv.json') as f:
    data = json.load(f)

# generate embeddings
lol = {}
import time
a = time.time()
for i, (key, val) in enumerate(data.items()):
    abstract = val['metadata']['entry']['summary']
    prompt = f'{abstract}'
    #output = llama(prompt)
    val['embedding'] = llama.embed(prompt)   
    lol[key] = val
    if i == 10:
        break

b = time.time()

print('EXECUTED IN', b-a)

with open('lol.json', 'w') as f:
    json.dump(lol, f, indent=4)

