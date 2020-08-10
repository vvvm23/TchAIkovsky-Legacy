import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import time
from datetime import timedelta
import random

import model
import dataloader

import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

NB_EPOCHS = 20

def train(model, dataloader):
    nb_batches = len(dataloader) 

    crit = nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    t_loss = 0.0

    for ei in range(NB_EPOCHS):
        for i, (batch_in, batch_out) in enumerate(dataloader):
            stime_batch = time.time()

            batch_in = batch_in.type(torch.LongTensor).to(device)
            batch_out = batch_out.type(torch.LongTensor).to(device)

            optim.zero_grad()
            out = model(batch_in)
            loss = crit(out, batch_out)
            loss.backward()
            optim.step()

            etime_batch = time.time()
            
            if not i % 32:
                estimated_time = timedelta(seconds = math.floor((etime_batch - stime_batch) * (nb_batches - i)))
                # estimated_time = estimated_time - timedelta(microseconds=estimated_time.microseconds)
                print(f"Epoch {ei+1}/{NB_EPOCHS} - Batch {i+1}/{nb_batches}")
                print(f"Batch finished in {etime_batch - stime_batch:1.2f} seconds")
                print(f"Estimated time to end of epoch: {str(estimated_time)}")
                print(f"{loss.item()}\n")

if __name__ == '__main__':
    transformer = model.TransformerModel(333, 128, 4, 256, 3, dropout=0.1, device=device).to(device)
    # transformer = nn.Sequential(
        # nn.Embedding(333, 256),
        # model.PositionalEncoding(256, dropout=0.1),
        # nn.Transformer(256, 8, 6, 6, 512, dropout=0.1)
    # )
    
    y = dataloader.MusicDataset("./np_out", device=device)

    dataloader = torch.utils.data.DataLoader(y, batch_size=32, shuffle=True, num_workers=0)

    print(torch.cuda.is_available())
    train(transformer, dataloader)
