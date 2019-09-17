import keras
from keras.models import load_model

import numpy as np

import music21
from music21.note import Note
from music21.chord import Chord
from music21.stream import Stream

from time import time

class MusicGenerator:
    def __init__(self, model_path=None, start_path=None):
        self.model_path = model_path
        self.start_path = start_path

        if model_path == None:
            print("Missing model path, please provide as arg in generate.")
        else:
            if _load_model():
                print(f"Failed to load model from path {self.model_path}.")
                exit()


        if start_path == None:
            print("Missing start path, please provide as arg in generate.")
        else:
            if _load_start():
                print(f"Failed to load start data from path {self.start_path}.")
                exit()

    def _load_model(self, model_path):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(e.message)
            return 0
        return 1

    def _load_start(self, start_path):
        try:
            start = np.load(start_path)
            self.start = start[:100, :]
        except Exception as e:
            print(e.message)
            return 0
        return 1

    def _generate_vectors(self, gen_len):
        ON_THRESHOLD = 0.8

        output = np.empty((gen_len, self.start.shape[1]))
        seed = self.start

        for i in range(len(gen_len)):
            output[i, :] = self.model.predict(seed, batch_size=1)
            for n in range(0, seed.shape[1], 2):
                output[i, n] = 1 if output[i, n] > ON_THRESHOLD else 0

            seed = np.append(seed[1:, :], output[i, :], axis=0)

        return output

    def _parse_vectors(self, vector_sequence):
        DUR_NORM = 4.0
        DUR_PREC = 0.1

        stream = Stream()

        for t in range(len(vector_sequence.shape[0])):
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
            _load_model()

        if self.start_path == None:
            self.start_path = start_path
            _load_start()

        vector_sequence = _generate_vectors(gen_len)
        note_stream = _parse_vectors(vector_sequence)
        note_stream.write('midi', f"./music/{int(time())}")