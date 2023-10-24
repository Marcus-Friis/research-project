from llama_cpp import Llama

model_path = 'openbuddy-llama2-13b-v11.1.Q5_K_M.gguf'
n_gpu_layers = -1
embedding = True
verbose = False
llama = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, embedding=embedding, verbose=verbose)


prompt = 'Q: who is the most beautiful? Mads or Morten? A:'

output = llama(prompt)

print(output["choices"][0]["text"])

