import numpy as np
import keras
import pandas as pd

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dim, ID_list, batch_size=32, seq_size=100, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seq_size = seq_size
        self.ID_list = ID_list
        self.on_epoch_end()

    '''def __ID_list_generation(self, meta_file):
        df = pd.read_csv(meta_file, header=None, usecols=[0,1])
        fl_pairs = list(zip(df[0].tolist(), df[1].tolist()))
        _ID_list = []

        for pair in fl_pairs:
            _ID_list = _ID_list + [(pair[0], i) for i in range(pair[1] - self.seq_size + 1)]

        return _ID_list'''

    def __len__(self):
        return int(np.floor(len(self.ID_list) / self.batch_size))

    def __getitem__(self, index):
        _indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        _ID_list = [self.ID_list[i] for i in _indexes]
        X, Y = self.__data_generation(_ID_list)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ID_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, _ID_list):
        X = np.empty((self.batch_size, self.seq_size, self.dim))
        Y = np.empty((self.batch_size, self.dim))

        for i, ID in enumerate(_ID_list):
            file_name = ID[0]
            start = ID[1]
            
            midi_vectors = np.load(file_name, mmap_mode='r') 
            X[i, :] = midi_vectors[start:start+self.seq_size, :]
            Y[i, :] = midi_vectors[start+self.seq_size, :]

        return X, Y



