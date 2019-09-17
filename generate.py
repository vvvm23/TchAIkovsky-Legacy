import keras
from keras.models import load_model

import numpy as np

import music21
from music21.note import Note
from music21.chord import Chord
from music21.stream import Stream

from time import time

from tqdm import tqdm

class MusicGenerator:
    def __init__(self, model_path=None, start_path=None):
        self.model_path = model_path
        self.start_path = start_path

        if model_path == None:
            print("Missing model path, please provide as arg in generate.")
        else:
            if self._load_model():
                print(f"Failed to load model from path {self.model_path}.")
                exit()


        if start_path == None:
            print("Missing start path, please provide as arg in generate.")
        else:
            if self._load_start():
                print(f"Failed to load start data from path {self.start_path}.")
                exit()

    def _load_model(self):
        try:
            self.model = load_model(self.model_path)
        except:
            return 1
        return 0

    def _load_start(self):
        try:
            start = np.load(self.start_path)
            self.start = start[:100, :]
        except:
            return 1
        return 0

    # TODO: Remove 'magic' numbers
    def _generate_vectors(self, gen_len):
        ON_THRESHOLD = 0.8

        output = np.empty((gen_len, self.start.shape[1]))
        seed = self.start

        for i in tqdm(range(gen_len)):
            output[i, :] = self.model.predict(seed.reshape(1, 100, 176), batch_size=1).reshape(176)
            for n in range(0, seed.shape[1], 2):
                output[i, n] = 1 if output[i, n] > ON_THRESHOLD else 0

            #print(seed[1:, :].shape, output[i, :].shape)
            seed = np.append(seed[1:, :], output[i, :].reshape(1, 176), axis=0)

        return output

    def _parse_vectors(self, vector_sequence):
        DUR_NORM = 4.0
        DUR_PREC = 0.1

        stream = Stream()

        for t in tqdm(range(vector_sequence.shape[0])):
            offset = DUR_PREC * t
            notes_vec = []

            for i in range(0, vector_sequence.shape[1], 2):
                if vector_sequence[t, i] == 1:
                    notes_vec.append((1, vector_sequence[t, i+1] * DUR_NORM))

            if len(notes_vec):
                # Maybe round here?
                notes = [Note(m, quarterLength=round(l, 2)) for m,l in notes_vec]
                chord = Chord(notes)
                stream.append(chord)
                stream[-1].offset = offset

        return stream

    def generate(self, gen_len=1000, model_path=None, start_path=None):
        if self.model_path == None and model_path == None:
            print("No model path defined.")
            exit()

        if self.start_path == None and start_path == None:
            print("No start path defined.")
            exit()

        if self.model_path == None:
            self.model_path = model_path
            self._load_model() # Do checking here

        if self.start_path == None:
            self.start_path = start_path
            self._load_start() # And here

        print("Generating Music Vectors")
        vector_sequence = self._generate_vectors(gen_len)

        print("Parsing Music Vectors")
        note_stream = self._parse_vectors(vector_sequence)

        save_name = int(time())
        print(f"Saving to file ./music/{save_name}.midi")
        note_stream.write('midi', f"./music/{save_name}.midi")
        #note_stream.write('midi')

generator = MusicGenerator("./models/tchAIkovsky-1568651365-03.h5", "./preprocessing/np_out/0.npy")
generator.generate(gen_len=5000)