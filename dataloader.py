import torch
import numpy as np
import glob

INTERVAL = 16
SAMPLE_LENGTH = 256

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.index_lookup = None
        self.music_sequences = {}

        self._generate_ids()

    def _generate_ids(self):
        INTERVAL = 16
        npy_files = glob.glob(f"{self.root_dir}/*.npy")
        self.index_lookup = {}
        sample_count = 0

        for file_id, path in enumerate(npy_files):
            seq = np.load(path)
            seq_len = seq.shape[0]

            self.music_sequences[file_id] = seq
            self.index_lookup.update({sample_count+i: (file_id, i*INTERVAL) for i in range((seq_len - SAMPLE_LENGTH) // INTERVAL) })
            sample_count += (seq_len - SAMPLE_LENGTH) // INTERVAL
            
    def __len__(self):
        return len(self.index_lookup)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_len = len(idx)
        out = np.zeros((idx_len, SAMPLE_LENGTH))
    
        for ei, i in enumerate(idx):
            file_id, loc = self.index_lookup[i] # (file_id, location)
            seq = self.music_sequences[file_id][loc:loc+SAMPLE_LENGTH]
            out[ei, :] = seq

        return out
