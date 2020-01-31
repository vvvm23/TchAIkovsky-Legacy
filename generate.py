import keras
from keras.models import load_model

import numpy as np
from time import time
from tqdm import tqdm

'''
    1) Load model from file
    2) Load starting music (or generate random?)
    3) Query network gen_len times
        - At end of every query, pop off old 1st and add new to end (with argmax)
        - Append result to output vector sequence (int seq, not vector)
    4) Take int sequence and convert to corresponding MIDI instruction
        - If invalid, still add. Csvmidi will handle
        - Record current time and velocity and advance as needed
    5) Save to csv file (or pass stdin to Csvmidi.exe directly?)
    6) Invoke Csvmidi.exe on file.
'''

class MusicGenerator:
    def __init__(self):
        pass

    def _load_starter(self, start_path):
        pass

    def _query_model(self, gen_len):
        pass

    def _seq_to_midi():
        pass

    def _save_to_csv():
        pass

    def _convert_csv():
        pass

    def generate(self):
        pass