from openai import AsyncOpenAI as OpenAI, RateLimitError, APIConnectionError
import json
from configparser import ConfigParser
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random, retry_if_exception_type, RetryError
import random


async def prompt(text, sleep=0):
    await asyncio.sleep(sleep)
    print(text)
    return text

async def batch_prompt(*args):
    return await asyncio.gather(*[prompt(arg[0], sleep=arg[1]) for arg in args])


class Chad:
    def __init__(self, 
                 model='gpt-3.5-turbo',
                 wait_fixed=1,
                 stop_after_attempt=10) -> None:
        config = ConfigParser()
        config.read('config.ini')
        api_key = config['openai']['api_key']
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.wait_fixed = wait_fixed
        self.stop_after_attempt = stop_after_attempt
    
    async def async_prompt(self, text, context='', delay=0, identifier=None):
        await asyncio.sleep(delay)
        
        @retry(stop=stop_after_attempt(self.stop_after_attempt), 
               wait=wait_fixed(self.wait_fixed) + wait_random(0, 10),
               retry=(retry_if_exception_type(RateLimitError) | retry_if_exception_type(APIConnectionError)))
        async def wrapper():
            print(f'prompting {identifier}')
            messages = [
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': text}
                ]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=10
                )
            print(f'done {identifier}')
            return response
        try:
            return await wrapper()
        except RetryError:
            print(f'failed {identifier}')
            return None
    
    def prompt(self, text, context='', sleep=0):
        return asyncio.run(self.async_prompt(text, context=context, sleep=sleep))
    
    async def async_batch_prompt(self, *args):
        return await asyncio.gather(*[self.async_prompt(*arg) for arg in args])
    
    def batch_prompt(self, *args):
        return asyncio.run(self.async_batch_prompt(*args))


if __name__ == '__main__':
    import json
    import time
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('start', type=int, help='Start value of the range', default=0, nargs='?')
    parser.add_argument('end', type=int, help='End value of the range', default=-1, nargs='?')

    args = parser.parse_args()

    start = args.start
    end = args.end
    
    with open('../data/arxiv.json', 'r') as f:
        arxiv = json.load(f)
        
    with open('../data/Cit-HepPh.txt', 'r') as f:
        for _ in range(4):
            next(f)
        edges = f.readlines()
    edges = [edge.strip().split('\t') for edge in edges]
    edges = edges[start:end]
    
    chad = Chad()
    
    prompts = []
    
    for i, (v, u) in enumerate(edges):
        try:
            abstract_v = arxiv[v]['metadata']['entry']['summary']
            abstract_u = arxiv[u]['metadata']['entry']['summary']
        except KeyError:
            continue
        text = f"""
        Given the following abstract: {abstract_v}
        How does it relate to this abstract: {abstract_u}
        Which of the following categories does it belong to: Agreement, Disagreement, Neutral.
        Describe it using strictly the category
        """
        context = ''
        delay = 1*i
        identifier = f'{v} {u}'
        entry = (text, context, delay, identifier)
        prompts.append(entry)
        
    a = time.time()
    out = chad.batch_prompt(*prompts)
    print('EXECUTED IN:\t', time.time() - a)
    
    import pickle
    with open(f'../data/chad_{start}_{end}.pkl', 'wb') as f:
        pickle.dump(out, f)
    
    
    # HOW TO UNPICKLE AND INTERPRET
    # import pickle
    
    # with open('../data/chad.pkl', 'rb') as f:
    #     response = pickle.load(f)
    
    # print(response)
    # print([r.choices[0].message.content for r in response])
