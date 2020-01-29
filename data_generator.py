import numpy as np
import keras

# Data Generator for handling our converted MIDI files
class DataGenerator(keras.utils.Sequence):
    def __init__(self, dim, seq_size, ID_list, batch_size=32, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seq_size = seq_size
        self.ID_list = ID_list
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.ID_list) / float(self.batch_size)))

    # Get one batch of data
    def __getitem__(self, index):
        if len(self.ID_list) - index*self.batch_size < self.batch_size:
            _indexes = self.indexes[index*self.batch_size:]
        else:
            _indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        _ID_list = [self.ID_list[i] for i in _indexes]
        X, Y = self.__data_generation(_ID_list)
        return X, Y

    # Called at end of epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ID_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Load the data from file and memory map it
    def __data_generation(self, _ID_list):
        X = np.zeros((len(_ID_list), self.seq_size, self.dim))
        Y = np.zeros((len(_ID_list), self.dim))

        for i, ID in enumerate(_ID_list):
            file_name = ID[0]
            start = ID[1]
            
            midi_vectors = np.load(file_name, mmap_mode='r') 
            X[i, :, :] = midi_vectors[start:start+self.seq_size, :] # Convert from token ids to argmax tokens
            Y[i, :] = midi_vectors[start+self.seq_size, :]

        return X, Y
