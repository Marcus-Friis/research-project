from llama_cpp import Llama

# prepare model
model_path = 'openbuddy-llama2-13b-v11.1.Q5_K_M.gguf'
n_gpu_layers = -1
embedding = True
verbose = False
llama = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, embedding=embedding, verbose=verbose)

# define prompt
prompt = 'Q: who is the most beautiful? Mads or Morten? A:'

# generate text
output = llama(prompt)
print(output)

# generate embedding
embedding = llama.embed(prompt)
print(embedding)

