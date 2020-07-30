import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import time
import random

import model
import dataloader

device = torch.cuda.device('cuda' if torch.cuda.is_available() else 'cpu')

NB_EPOCHS = 20

def train(model):
    model.train()
    t_loss = 0.0
    s_time = time.time()

    for ei in range(NB_EPOCHS):
        pass

if __name__ == '__main__':
    x = model.TransformerModel(256, 256, 8, 512, 6, dropout=0.1)
    print(x)

    y = dataloader.MusicDataset("./np_out")
    print(y.__getitem__(random.sample(range(y.__len__()), 5)))
