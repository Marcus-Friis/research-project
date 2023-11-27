from chatgpt import Chad

if __name__ == '__main__':
    import json
    import numpy as np
    import pickle
    
    with open('../data/arxiv.json', 'r') as f:
        arxiv = json.load(f)
        
    with open('../data/Cit-HepPh.txt', 'r') as f:
        for _ in range(4):
            next(f)
        edges = f.readlines()
    edges = [edge.strip().split('\t') for edge in edges]
    
    start = 0
    end = len(edges)
    size = 10
    np.random.seed(42)
    idx = np.random.uniform(start, end, size=size).astype(int)
    edges = np.array(edges)
    edges = edges[idx]
    
    prompts = []
    for i, (v, u) in enumerate(edges):
        try:
            abstract_v = arxiv[v]['metadata']['entry']['summary']
            abstract_u = arxiv[u]['metadata']['entry']['summary']
        except KeyError:
            continue
        text = f"""
        Paper A: {abstract_v}
        Paper B: {abstract_u}
        """
        context = """
        You are a high energy physics expert.
        You will get the abstract of a paper that cites another paper within the field of high energy physics.
        You will evaluate to the best of your ability, whether the paper agrees with the other paper.
        Paper A cites paper B.
        The abstract of paper A will follow after "Paper A:", and the abstract of paper B will follow after "Paper B:"
        You will only provide single word answer, evaluating the agreement between the papers.
        Use "agreement" when Paper A clearly agrees with or builds upon the conclusions of Paper B.
        Use "disagreement" when Paper A clearly contradicts or refutes the conclusions of Paper B.
        Use "neutral" when Paper A is neutral to the conclusions of Paper B or if it is unclear how the papers relate.
        """
        delay = .1*i
        identifier = f'{v} {u}'
        entry = (text, context, delay, identifier)
        prompts.append(entry)

    chad = Chad(wait_fixed=10)
    x = []
    for N in range(5):
        out = chad.batch_prompt(*prompts)
        x.append(out)
        
    with open(f'../data/chad.pkl', 'wb') as f:
        pickle.dump(x, f)

    # with open(f'../data/chad.pkl', 'rb') as f:
    #     x = pickle.load(f)
    
    new_x = []
    for z in x:
        inner_list = []
        for r in z:
            inner_list.append(r.choices[0].message.content)
        new_x.append(inner_list)
    x = new_x
            
    print(np.array(x).T)