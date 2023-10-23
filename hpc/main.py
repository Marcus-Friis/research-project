from llama_cpp import Llama

model_path = 'openbuddy-llama2-13b-v11.1.Q5_K_M.gguf'
n_gpu_layers = -1
embedding = True
verbose = False
llama = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, embedding=embedding, verbose=verbose)


prompt = b'who is the most beautiful? Mads or Morten?'

print(llama.embed(prompt))

# tokens = llama.tokenize(prompt)
# for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1):
#      print(llama.detokenize([token]))

# response = llama.generate(prompt)
# print(response)