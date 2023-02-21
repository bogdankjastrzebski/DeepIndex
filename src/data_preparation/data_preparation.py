import pandas as pd
import time

def indexed_abstract_to_string(indexed_abstract):
    words = ["" for _ in range(indexed_abstract['IndexLength'])]
    for k, v in indexed_abstract['InvertedIndex'].items():
        for i in v: 
            words[i] = k
    return " ".join(words)

def authors_to_ids(authors):
    return [e['id'] for e in authors]

print("Load data...")

name = "dblp.v12"
path = "../../data"

cols = ['id', 'title', 'indexed_abstract', 'authors', 'references']

df = pd.read_json(f'{path}/{name}.json')

print("Filter...") 
df = df[~df[cols].isna().any(axis=1)]

print("Abstract...")
df['abstract']   = df.indexed_abstract.map(indexed_abstract_to_string)

print("Author ids...")
df['author_ids'] = df.authors.map(authors_to_ids)

print("Saving...")
df[['id', 'author_ids', 'title', 'abstract', 'references']].to_parquet(f'{path}/{name}.parquet')
