# research-project

By Mads Høgenhaug, Marcus Friis, Morten Pedersen.

## Data

The data used in this project is the *[Arxiv HEP-PH (high energy physics phenomenology) citation graph](https://snap.stanford.edu/data/cit-HepPh.html)*.

The embedded data: <https://www.mediafire.com/file/im0jf93tdihemrg/combined_data.rar/file>

## Used llama model

<https://huggingface.co/TheBloke/OpenBuddy-Llama2-13B-v11.1-GGUF>
openbuddy-llama2-13b-v11.1.Q5_K_M.gguf

## TODO

* ~~add embedding 191 nan~~
* do clustering
  * try more $k$ values
  * evaluate clustering with silhouette and/or H-plus and modularity when applied to graph
  * evaluate community with same as above
* compare clustering and community detection
* openai api access / script
  * COOK!
  * Evaluate stability with prompt engineering
  * Run gpt 3.5 and 4 on 100 samples multiple times, evaluate stability
  * compare with expert
* multi-layer community detection and compare
  * evaluate embedding distance between agreement, neutral and disagreennt edges
* write readme
* DONE!
