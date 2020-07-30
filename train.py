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

def train(model, dataloader):
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    t_loss = 0.0
    s_time = time.time()

    for ei in range(NB_EPOCHS):
        for batch_in, batch_out in dataloader:
            optim.zero_grad()
            # out = model(batch)
            print(batch_in.shape, batch_out.shape)

if __name__ == '__main__':
    x = model.TransformerModel(333, 128, 8, 512, 6, dropout=0.1, device=device)

    y = dataloader.MusicDataset("./np_out")

    dataloader = torch.utils.data.DataLoader(y, batch_size=32, shuffle=True, num_workers=8)

    train(x, dataloader)
