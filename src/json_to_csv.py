import pandas as pd
import json

def json_to_df(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    authors = []
    titles = []
    publish_dates = []
    abstracts = []
    for key in data.keys():
        try:
            author_list = data[key]['metadata']['entry']['author']
            authors.append(', '.join([author['name'] for author in author_list]))
        except:
            authors.append(None)
        try:
            titles.append(data[key]['metadata']['entry']['title'])
        except:
            titles.append(None)
        try:
            publish_dates.append(data[key]['metadata']['entry']['published'])
        except:
            publish_dates.append(None)
        try:
            abstracts.append(data[key]['metadata']['entry']['summary'])
        except:
            abstracts.append(None)

    df = pd.DataFrame({'id': data.keys(), 'author': authors, 'title': titles, 'publish_date': publish_dates, 'abstract': abstracts})
    return df

if __name__ == '__main__':
    json_file = '../data/arxiv.json'
    df = json_to_df(json_file)
    print(df)
    df.to_csv('../data/arxiv.csv', index=False)
