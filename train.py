import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import time
from datetime import timedelta
import random

import model
import dataloader
import midigen

import math

TRY_CUDA = True
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'}) \n")

NB_EPOCHS = 500
PRINT_INV = 64

def train(model, dataloader):
    nb_batches = len(dataloader) 

    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)

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
            out = out.transpose(1, 2)

            loss = crit(out, batch_out)
            total_loss += loss.item()
            loss.backward()

            optim.step()
            # scheduler.step()

            etime_batch = time.time()
            
            if not i % PRINT_INV:
                estimated_time = timedelta(seconds = math.floor((etime_batch - stime_batch) * (nb_batches - i)))
                print(f"Epoch {ei+1}/{NB_EPOCHS} - Batch {i+1}/{nb_batches}")
                print(f"Batch finished in {etime_batch - stime_batch:1.2f} seconds")
                print(f"Estimated time to end of epoch: {str(estimated_time)}")
                print(f"Loss: {total_loss / PRINT_INV}\n")
                total_loss = 0.0

    torch.save(model, f"models/{int(time.time())}-model.pt")
    generate(model, f"{ei}-sample.csv", primer=test_primer)

def generate(model, name, primer=None):
    model.eval()
    primer_length = 512

    EOS_TOKEN = 334
    MAX_LENGTH = 5000

    if primer == None:
        primer = torch.tensor([random.randint(0, 332) for _ in range(primer_length)]).to(device)
        primer = primer.view(1, -1)
    
    primer = primer.type(torch.LongTensor).to(device)
    # sample = primer.reshape(primer_length).tolist()
    sample = []

    print("> Generating Sample.")
    while len(sample) < MAX_LENGTH:
        print(f"> Sample Length: {len(sample)}")
        out = model(primer)
        out = torch.argmax(out, dim=-1)
        sample = sample + out.reshape(primer_length).tolist()

        if EOS_TOKEN in out.reshape(primer_length).tolist():
            break

        primer = out

    midigen.generate_from_seq(sample, f"samples/{name}")

    model.train()
    
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'}) \n")

    print("> Using Tensorflow Magenta MIDI Dataset\n")
    y = dataloader.MusicDataset("./np_out")
    test_primer = y.__getitem__(0)[0].type(torch.LongTensor).view(1, -1)
    dataloader = torch.utils.data.DataLoader(y, batch_size=16, shuffle=True, num_workers=8)

    transformer = model.TransformerModel(335, 256, 8, 512, 6, device=device).to(device)
    print("> Model Summary:")
    print(transformer, '\n')

    # model = torch.load("models/1598287965-model.pt")
    # generate(model, "load-test")

    train(transformer, dataloader)
