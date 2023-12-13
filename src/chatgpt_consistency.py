from chatgpt import Chad


context_dict = {
    'baseline': """
    You are given two abstracts; one from Paper A and one from Paper B.
    Paper A cites Paper B. Give a single word answer, evaluating the agreement between the papers. 
    Use one of these labels: agreement, disagreement, neutral.
    """,
    
    'simple': """
    You are a high energy physics expert.
    You will get the abstract of a paper that cites another paper within the field of high energy physics.
    You will evaluate to the best of your ability, whether the paper agrees with the other paper.
    Paper A cites paper B.
    The abstract of paper A will follow after "Paper A:", and the abstract of paper B will follow after "Paper B:"
    You will only provide single word answer, evaluating the agreement between the papers.    
    """,
    
    'medium': """
    You are a high energy physics expert.
    You will get the abstract of a paper that cites another paper within the field of high energy physics.
    You will evaluate to the best of your ability, whether the paper agrees with the other paper.
    Paper A cites paper B.
    The abstract of paper A will follow after "Paper A:", and the abstract of paper B will follow after "Paper B:"
    You will only provide single word answer, evaluating the agreement between the papers.
    Use "agreement" when Paper A agrees with Paper B.
    Use "disagreement" when Paper A contradicts Paper B.
    Use "neutral" when it is neither "agreement" or "disagreement".    
    """,
    
    'final': """
    You are a high energy physics expert.
    You will get the abstract of a paper that cites another paper within the field of high energy physics.
    You will evaluate to the best of your ability, whether the paper agrees with the other paper.
    Paper A cites paper B.
    The abstract of paper A will follow after "Paper A:", and the abstract of paper B will follow after "Paper B:"
    You will only provide single word answer, evaluating the agreement between the papers.
    Use "agreement" when Paper A clearly agrees with or builds upon the conclusions of Paper B.
    Use "disagreement" when Paper A clearly contradicts or refutes the conclusions of Paper B.
    Use "neutral" when Paper A is neutral to the conclusions of Paper B or if it is unclear how the papers relate.
    """,
    
    'explain': """
    You are a high energy physics expert. 
    You will get the abstract of a paper that cites another paper within the field of high energy physics. 
    You will evaluate to the best of your ability, whether the paper agrees with the other paper. 
    Paper A cites paper B. 
    The abstract of paper A will follow after "Paper A:", and the abstract of paper B will follow after "Paper B:" 
    You will only provide single word answer, evaluating the agreement between the papers, followed by an explanation of your reasoning. 
    Use "agreement" when Paper A clearly agrees with or builds upon the conclusions of Paper B. 
    Use "disagreement" when Paper A clearly contradicts or refutes the conclusions of Paper B. 
    Use "neutral" when Paper A is neutral to the conclusions of Paper B or if it is unclear how the papers relate.
    """
}


if __name__ == '__main__':
    import json
    import numpy as np
    import pickle
    import argparse
    import time
    
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--context', type=str, default='final', help='Context of the prompt, choose from' + ', '.join(context_dict.keys()))
    parser.add_argument('-m', '--model', type=str, default='gpt-3.5-turbo')
    
    args = parser.parse_args()
    context_label = args.context
    model = args.model
    
    with open('../data/arxiv.json', 'r') as f:
        arxiv = json.load(f)
        
    with open('../data/Cit-HepPh.txt', 'r') as f:
        for _ in range(4):
            next(f)
        edges = f.readlines()
    edges = [edge.strip().split('\t') for edge in edges]
    
    start = 0
    end = len(edges)
    size = 100
    np.random.seed(42)
    idx = np.random.uniform(start, end, size=size).astype(int)
    edges = np.array(edges)
    edges = edges[idx]
    
    prompts = []
    for i, (v, u) in enumerate(edges):
        abstract_v = arxiv[v]['metadata']['entry']['summary']
        abstract_u = arxiv[u]['metadata']['entry']['summary']
        
        text = f"""
        Paper A: {abstract_v}
        Paper B: {abstract_u}
        """
        context = context_dict[context_label]
        delay = .1*i
        identifier = f'{i}:\t{v} {u}'
        entry = (text, context, delay, identifier)
        prompts.append(entry)

    chad = Chad(wait_fixed=10, model=model)
    
    x = []
    for N in range(5):
        print('RUN', N)
        out = chad.batch_prompt(*prompts)
        x.append(out)
        
    with open(f'../data/prompt_engineering/{context_label}_{model}.pkl', 'wb') as f:
        pickle.dump(x, f)

    # with open(f'../data/prompt_engineering/{context_label}_{model}.pkl', 'rb') as f:
    #     x = pickle.load(f)
    
    new_x = []
    for z in x:
        inner_list = []
        for r in z:
            inner_list.append(r.choices[0].message.content)
        new_x.append(inner_list)
    x = new_x
            
    x = np.array(x).T
    x = np.char.lower(x)
    
    print(x)
    
    ## CODE FOR ANALYSIS
    # import pandas as pd
    # df = pd.DataFrame([' '.join(edge)for edge in edges], columns=['edge'])
    # df['labels'] = x.tolist()
    
    # def label_check(x):
    #     if 'disagree' in x:
    #         return -1
    #     if 'agree' in x:
    #         return 1
    #     if 'neutral' in x:
    #         return 0
    #     return np.nan
    
    # df['label_values'] = [[label_check(text) for text in row] for row in x]
    
    # df['mode'] = df.label_values.apply(lambda x: max(set(x), key=x.count))
    # df['std'] = df.label_values.apply(np.nanstd)
    
    # print(df)
    # print(df['std'].mean())
