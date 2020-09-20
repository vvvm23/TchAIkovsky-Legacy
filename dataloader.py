import torch
import numpy as np
import glob

from tqdm import tqdm

INTERVAL = 128
SAMPLE_LENGTH = 256
INITIAL_ALLOC = 300_000

class FastDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.src = torch.zeros((INITIAL_ALLOC, SAMPLE_LENGTH)).type(torch.LongTensor)
        self.tgt = torch.zeros((INITIAL_ALLOC, SAMPLE_LENGTH)).type(torch.LongTensor)
        self.out = torch.zeros((INITIAL_ALLOC, SAMPLE_LENGTH)).type(torch.LongTensor)

        self.length = 0

        self._load_data()

    def _load_data(self):
        npy_files = glob.glob(f"{self.root_dir}/*.npy")

        pb = tqdm(total=len(npy_files))

        for path in npy_files[:100]:
            full_seq = torch.from_numpy(np.load(path))
            full_seq_len = full_seq.shape[0]

            locs = range(0, full_seq_len - SAMPLE_LENGTH - 1, INTERVAL)

            for l in locs:
                seq = full_seq[l:l+SAMPLE_LENGTH*2]

                src = torch.zeros(SAMPLE_LENGTH).type(torch.LongTensor)
                tgt = torch.zeros(SAMPLE_LENGTH).type(torch.LongTensor)
                out = torch.zeros(SAMPLE_LENGTH).type(torch.LongTensor)

                src = seq[:SAMPLE_LENGTH]
                
                pad_len = 2*SAMPLE_LENGTH - seq.shape[0]

                if pad_len > 0:
                    out[:SAMPLE_LENGTH - pad_len] = seq[SAMPLE_LENGTH:SAMPLE_LENGTH*2]
                    out[SAMPLE_LENGTH - pad_len:] = torch.tensor([0]*pad_len)

                    tgt[0] = torch.tensor(1)
                    tgt[1:SAMPLE_LENGTH - pad_len] = seq[SAMPLE_LENGTH:-1]
                    tgt[SAMPLE_LENGTH - pad_len:] = torch.tensor([0]*(pad_len))
                else:
                    out = seq[SAMPLE_LENGTH:SAMPLE_LENGTH*2]

                    tgt[0] = torch.tensor(1)
                    tgt[1:] = seq[SAMPLE_LENGTH:SAMPLE_LENGTH*2 - 1]

                self.src[self.length, :] = src
                self.tgt[self.length, :] = tgt
                self.out[self.length, :] = out

                self.length += 1

            pb.update(1)
        
        self.src = self.src[:self.length, :]
        self.tgt = self.tgt[:self.length, :]
        self.out = self.out[:self.length, :]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.out[idx]
