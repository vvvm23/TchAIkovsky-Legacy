#import keras
#from keras.models import load_model

import numpy as np
from time import time
from tqdm import tqdm

import argparse

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
    def __init__(self, model_path):
        self.model = None #load_model(model_path)

    def _load_starter(self, start_path, start_length):
        midi_vectors = np.load(start_path)
        return midi_vectors[:start_length, :]


    def _query_model(self, gen_len):
        pass

    def _seq_to_midi(self):
        pass

    def _save_to_csv(self, csv_data, save_path):
        f = open(save_path, mode='w')
        f.writelines(csv_data)
        f.close()

    def _convert_csv_file(self, csv_path, save_path, csvmidi_path):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.system(f"{csvmidi_path} {csv_path} {save_path}")

    def generate(self, start_data=None, gen_len=1000):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generates new music from a previously trained model")
    parser.add_argument("model_path", type=str, help="Compulsory argument. Path to a saved model.")
    parser.add_argument("--csvmidi_path", type=str, help="Path to Csvmidi.exe. If on Unix, may need to recompile", default="preprocessing/Csvmidi.exe")
    parser.add_argument("--start_path", type=str, help="Path to starting data. If None, random?", default=None)
    parser.add_argument("--start_length", type=int, help="Length of starter data, should be equal to length of training examples. (?)", default=200)
    parser.add_argument("--gen_len", type=int, help="Length of the song to be generated in number samples.", default=1000)
    args = parser.parse_args()

    CSVMIDI_PATH = args.csvmidi_path

    generator = MusicGenerator(args.model_path)
    generator.generate(args.start_path, args.gen_len)
