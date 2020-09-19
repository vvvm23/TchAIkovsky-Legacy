import torch
import numpy as np
import glob

INTERVAL = 128
SAMPLE_LENGTH = 256

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.index_lookup = None
        self.music_sequences = {}

        self._generate_ids()

    def _generate_ids(self):
        npy_files = glob.glob(f"{self.root_dir}/*.npy")
        self.index_lookup = {}
        sample_count = 0

        for file_id, path in enumerate(npy_files[:1]):
            seq = np.load(path)
            seq_len = seq.shape[0]

            self.music_sequences[file_id] = torch.from_numpy(seq)
            self.index_lookup.update({sample_count+i: (file_id, i*INTERVAL) for i in range((seq_len - SAMPLE_LENGTH - 1) // INTERVAL) })
            sample_count += (seq_len - SAMPLE_LENGTH - 1) // INTERVAL
            
    def __len__(self):
        return len(self.index_lookup)

    def __getitem__(self, idx):
        src = torch.zeros(SAMPLE_LENGTH).type(torch.LongTensor)
        tgt = torch.zeros(SAMPLE_LENGTH).type(torch.LongTensor)
        out = torch.zeros(SAMPLE_LENGTH).type(torch.LongTensor)

        file_id, loc = self.index_lookup[idx]

        seq = self.music_sequences[file_id][loc:loc+SAMPLE_LENGTH*2]

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

        return src, tgt, out
