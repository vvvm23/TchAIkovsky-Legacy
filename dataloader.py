import torch
import numpy as np
import glob

INTERVAL = 64
SAMPLE_LENGTH = 128

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):#, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.root_dir = root_dir
        self.index_lookup = None
        self.music_sequences = {}
        # self.device = device

        self._generate_ids()

    def _generate_ids(self):
        # INTERVAL = 
        npy_files = glob.glob(f"{self.root_dir}/*.npy")
        self.index_lookup = {}
        sample_count = 0

        for file_id, path in enumerate(npy_files[:5]):
            seq = np.load(path)
            seq_len = seq.shape[0]

            self.music_sequences[file_id] = torch.from_numpy(seq)#.to(self.device)
            self.index_lookup.update({sample_count+i: (file_id, i*INTERVAL) for i in range((seq_len - 2*SAMPLE_LENGTH) // INTERVAL) })
            sample_count += (seq_len - 2*SAMPLE_LENGTH) // INTERVAL
            
    def __len__(self):
        return len(self.index_lookup)

    def __getitem__(self, idx):
        out = torch.zeros(SAMPLE_LENGTH)
        target = torch.zeros(SAMPLE_LENGTH)

        file_id, loc = self.index_lookup[idx]
        # seq = self.music_sequences[file_id][loc:loc+SAMPLE_LENGTH+1]
        # out[:] = seq[:SAMPLE_LENGTH]
        # target = seq[1:1+SAMPLE_LENGTH].view(-1)

        seq = self.music_sequences[file_id][loc:loc+SAMPLE_LENGTH*2]
        out[:] = seq[:SAMPLE_LENGTH]
        target = seq[SAMPLE_LENGTH:2*SAMPLE_LENGTH].view(-1)

        return (out, target)
