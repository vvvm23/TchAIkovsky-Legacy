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

import sys
import math
import numpy as np

TRY_CUDA = True

NB_EPOCHS = 1000
PRINT_INV = 64
GEN_INV = 10

RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.1

def train(model, train_dataloader, test_dataloader):
    run_id = int(time.time())
    nb_batches = len(train_dataloader) 

    # crit = nn.CrossEntropyLoss()
    crit = nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.25, patience=2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)

    model.train()

    for ei in range(NB_EPOCHS):
        epoch_loss = 0.0
        total_loss = 0.0
        for i, (batch_src, batch_tgt, batch_out) in enumerate(train_dataloader):

            stime_batch = time.time()

            batch_src = batch_src.type(torch.LongTensor).to(device)
            batch_tgt = batch_tgt.type(torch.LongTensor).to(device)
            batch_out = batch_out.type(torch.LongTensor).to(device)

            optim.zero_grad()
            out = model(batch_src, batch_tgt)

            # print(batch_src[0], batch_tgt[0])
            # print(batch_out[0], out[0])

            out = out.reshape(-1, 336)
            batch_out = batch_out.reshape(-1)

            loss = crit(out, batch_out)
            total_loss += loss.item()
            epoch_loss += loss.item()
            loss.backward()

            optim.step()
            # scheduler.step()

            etime_batch = time.time()
            
            if not i % PRINT_INV:
                estimated_time = timedelta(seconds = math.floor((etime_batch - stime_batch) * (nb_batches - i)))
                print(f"> Epoch {ei+1}/{NB_EPOCHS} - Batch {i+1}/{nb_batches}")
                print(f"> Batch finished in {etime_batch - stime_batch:1.2f} seconds")
                print(f"> Estimated time to end of epoch: {str(estimated_time)}")
                print(f"> Loss: {total_loss / PRINT_INV}\n")
                total_loss = 0.0

        # scheduler.step(epoch_loss / len(dataloader))

        if ei and not ei % GEN_INV:
            generate(model, f"{ei}-sample.csv", src=test_primer)

        val_loss = evaluate(model, test_dataloader, optim, crit)
        print(f"> Epoch {ei+1}/{NB_EPOCHS}")
        print(f"> Validation Loss: {val_loss}\n")

        torch.save(model, f"models/{run_id}-{ei}-model.pt")
    torch.save(model, f"models/{run_id}-final-model.pt")

def evaluate(model, dataloader, optim, crit):
    total_loss = 0.0
    model.eval()
    for batch_src, batch_tgt, batch_out in dataloader:
        batch_src = batch_src.type(torch.LongTensor).to(device)
        batch_tgt = batch_tgt.type(torch.LongTensor).to(device)
        batch_out = batch_out.type(torch.LongTensor).to(device)

        optim.zero_grad()
        out = model(batch_src, batch_tgt)

        out = out.reshape(-1, 336)
        batch_out = batch_out.reshape(-1)

        loss = crit(out, batch_out)
        total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)

def generate(model, name, src=None):
    model.eval()
    src_len = 256

    EOS_TOKEN = 2
    MAX_LENGTH = 10000

    if src == None:
        src = torch.tensor([random.randint(3, 336) for _ in range(src_len)]).to(device)
        src = src.view(1, -1)
    

    src = src.type(torch.LongTensor).to(device)
    # sample = primer.reshape(primer_length).tolist()
    sample = []

    print("> Generating Sample.")
    while len(sample) < MAX_LENGTH:
        print(f"> Sample Length: {len(sample)}")
        tgt = torch.tensor([1] + [0]*(src_len - 1)).unsqueeze(0)
        tgt = tgt.type(torch.LongTensor).to(device)

        sub_sample = torch.zeros_like(src).to(device)

        for i in range(src_len):
            out = model(src, tgt)
            argmax_out = torch.argmax(out[0, i, :], dim=-1)

            sub_sample[0, i] = argmax_out
            if not i == src_len - 1:
                tgt[0, i+1] = argmax_out

        sample = sample + sub_sample.reshape(src_len).tolist()

        if EOS_TOKEN in sub_sample:
            break

        src = sub_sample

    for i, e in enumerate(sample):
        if e == EOS_TOKEN:
            sample = sample[:i-1]
            break

    midigen.generate_from_seq(sample, f"samples/{name}")

    model.train()
    
if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'}) \n")

    print("> Loading Tensorflow Magenta MIDI Dataset.")
    dataset = dataloader.FastDataset("./np_out")

    sample_indices = list(range(len(dataset)))
    np.random.shuffle(sample_indices)
    split_point = int(np.floor(TRAIN_TEST_SPLIT * dataset.__len__()))
    test_indices, train_indices = sample_indices[:split_point], sample_indices[split_point:]

    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    test_primer = dataset.__getitem__(test_indices[0])[0].type(torch.LongTensor).view(1, -1)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=8, sampler=train_sampler)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=8, sampler=test_sampler)
    print("> Done.")
    print(f"> Loaded {dataset.length} MIDI sequences.")

    transformer = model.TransformerModel(336, 128, 8, 256, 4, dropout=0.2, device=device).to(device)
    print("> Model Summary:")
    print(transformer, '\n')

    if len(sys.argv) == 2:
        print("> Loading existing model from file\n")
        transformer = torch.load(sys.argv[1])
    # generate(transformer, "load-test")

    train(transformer, train_dataloader, test_dataloader)
