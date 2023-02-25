import numpy as np
import torch
import time
import os

from lib.word2vec import *
from lib.utils import *

# --------------------------------------------------------------------------------- # 
#                                  Training                                         # 
# --------------------------------------------------------------------------------- #

path = '../../data'
name = 'articles_prepared'

model_path = "./models/0"
model_name = ""

X, transform = torch.load(f'{path}/{name}.pt')

X = X.cuda()

TRAIN_SIZE = X.shape[0]
VALID_SIZE = n // 100
NUMBER_UNIQUE = X.unique().shape[0]

dl_train, dl_valid, dl_test, ds_neg = prepare_dataloaders(X, 2*1024, [n - 2*m, m, m]) 

m, opt, losses = create_model(NUMBER_UNIQUE, 256)

if model_name:
    sd, opt, losses = torch.load(f'{model_path}/{model_name}.pt')
    m.load_state_dict(sd)

opt = torch.optim.SparseAdam(m.parameters(), lr=0.001)

train_model(m, opt, dl_train, dl_valid, losses, EPOCHS=10)

torch.save((m.state_dict(), opt, losses), f"{model_path}/model_final_{round(time.time()):012d}.pt")


