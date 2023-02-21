import pandas as pd
import numpy as np
import torch 

name = "dblp.v12"
path = "../../data"

print("Load data...")
df = pd.read_parquet(f"{path}/{name}.parquet")

print("Calculate unique ids...")
unique_ids = set(df.id.unique())

print("Compute observations...")
observations = (
    df
    .pipe(
        lambda x: x.assign(
            refs=x.references.map(lambda x: [x for x in x if x in unique_ids])
        )
    )
    .apply(
        lambda x:
           [(int(x['id']), e) for e in x['refs']]
           if ~np.isnan(x['refs']).any() and len(x['refs']) > 0
           else float('nan'),
        axis=1
    )
    .pipe(lambda x: x[~x.isna()])
    .pipe(lambda x: np.concatenate(x.values))
)

observations = torch.tensor(observations)

torch.save(observations, f'{path}/articles.pt')
