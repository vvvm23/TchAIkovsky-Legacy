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

NB_EPOCHS = 200
PRINT_INV = 32

def train(model, dataloader):
    nb_batches = len(dataloader) 

    crit = nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    t_loss = 0.0

    for ei in range(NB_EPOCHS):
        total_loss = 0.0
        for i, (batch_in, batch_out) in enumerate(dataloader):
            stime_batch = time.time()

            batch_in = batch_in.type(torch.LongTensor).to(device)
            batch_out = batch_out.type(torch.LongTensor).to(device)

            optim.zero_grad()
            out = model(batch_in)

            loss = crit(out, batch_out)
            total_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()

            etime_batch = time.time()
            
            if not i % PRINT_INV:
                estimated_time = timedelta(seconds = math.floor((etime_batch - stime_batch) * (nb_batches - i)))
                # estimated_time = estimated_time - timedelta(microseconds=estimated_time.microseconds)
                print(f"Epoch {ei+1}/{NB_EPOCHS} - Batch {i+1}/{nb_batches}")
                print(f"Batch finished in {etime_batch - stime_batch:1.2f} seconds")
                print(f"Estimated time to end of epoch: {str(estimated_time)}")
                print(f"Loss: {total_loss / PRINT_INV}\n")
                total_loss = 0.0

def generate(model):
    model.eval()
    primer_length = 256
    sample_length = 1024
    sample = []
    primer = torch.tensor([random.randint(0, 332) for _ in range(primer_length)]).to(device)
    primer = primer.view(1, -1)

    for i in range(sample_length): 
        out = model(primer)
        out = torch.argmax(out)
        sample.append(int(out))
        primer = torch.cat([primer[:, 1:], torch.reshape(out, (1, 1))], dim=1)

    # TODO: pass sample to midi generator

    model.train()
    
if __name__ == '__main__':
    transformer = model.TransformerModel(333, 128, 8, 512, 6, dropout=0.5, device=device).to(device)
    # transformer = nn.Sequential(
        # nn.Embedding(333, 256),
        # model.PositionalEncoding(256, dropout=0.1),
        # nn.Transformer(256, 8, 6, 6, 512, dropout=0.1)
    # )
    
    y = dataloader.MusicDataset("./np_out", device=device)

    dataloader = torch.utils.data.DataLoader(y, batch_size=32, shuffle=True, num_workers=0)

    print(torch.cuda.is_available())
    generate(transformer)
    # train(transformer, dataloader)
