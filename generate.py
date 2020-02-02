#import keras
#from keras.models import load_model
import tensorflow as tf

import numpy as np
from time import time
from tqdm import tqdm

import argparse

import time
import os
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
    def __init__(self, model_path, csvmidi_path):
        self.model = tf.keras.models.load_model(model_path)
        self.csvmidi_path = csvmidi_path

    def _load_starter(self, start_path, start_length):
        midi_vectors = np.load(start_path)
        return midi_vectors[:start_length, :]

    def _query_model(self, gen_len, start_data):
        output = []
        for i in range(gen_len):
            next_step = self.model.predict(np.reshape(start_data, (1, *start_data.shape)), batch_size=1)
            next_int = np.argmax(next_step)
            output.append(next_int)
            start_data = np.concatenate((start_data[1:, :], np.zeros((1, 317))), axis=0)
            start_data[-1, next_int] = 1.0
        return output

    def _seq_to_midi(self, seq):
        TRACK = "2, "
        NOTE_STUB = ", Note_on_c, 0, " # TRACK TIME NOTE_STUB NOTE_VAL VELOCITY
        
        # Begin by creating Header data
        midi_commands = [
            "0, 0, Header, 1, 2, 480",
            "1, 0, Start_track",
            "1, 0, Tempo, 500000",
            "1, 0, Time_signature, 4, 2, 24, 8",
            "1, 1, End_track",
            "2, 0, Start_track"
        ]

        MIN_NOTE = 21
        MAX_NOTE = 108
        MAX_TIME = 1000
        TIME_INTERVAL = 8
        MIN_VEL = 16
        MAX_VEL = 128
        VEL_INTERVAL = 8

        NB_NOTES = MAX_NOTE - MIN_NOTE + 1
        NB_TIME = MAX_TIME // TIME_INTERVAL + 1
        NB_VEL = (MAX_VEL - MIN_VEL) // VEL_INTERVAL + 1

        ON_START = 0
        OFF_START = NB_NOTES
        TIME_START = 2*NB_NOTES
        VEL_START = 2*NB_NOTES + NB_TIME

        current_time = 0
        current_velocity = 0

        for event in seq:
            if event < OFF_START: # Is ON event
                relative_note = event + MIN_NOTE
                midi_commands.append(f"{TRACK}{current_time}{NOTE_STUB}{relative_note}, {current_velocity}")
            elif event < TIME_START: # Is OFF event
                relative_note = event - OFF_START + MIN_NOTE
                midi_commands.append(f"{TRACK}{current_time}{NOTE_STUB}{relative_note}, 0")
            elif event < VEL_START: # Is TIME event
                # wait, is there a point in the first time vector?
                current_time += TIME_INTERVAL * (event - TIME_START + 1)
            else: # Is VEL event
                current_velocity = VEL_INTERVAL * (event - VEL_START) + MIN_VEL

        midi_commands.append(f"{TRACK}{current_time+1}, End_track")
        midi_commands.append(f"0, 0, End_of_file")
        return [x + '\n' for x in midi_commands]

    def _save_to_csv(self, csv_data, save_path):
        f = open(save_path, mode='w')
        f.writelines(csv_data)
        f.close()

    def _convert_csv_file(self, csv_path, save_path):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.system(f"{self.csvmidi_path} {csv_path} {save_path}")

    def generate(self, start_data=None, start_data_len=200, gen_len=1000):
        print("Starting Generation..")

        print("Loading start data.. ", end='', flush=True)
        start_data = self._load_starter(start_data, start_data_len)
        print("Done.")

        print("Querying model.. ", end='', flush=True)
        seq = self._query_model(gen_len, start_data)
        print("Done.")

        print("Translation to MIDI commands.. ", end='', flush=True)
        midi_commands = self._seq_to_midi(seq)
        print("Done.")

        save_id = f"{int(time.time())}"
        print("Saving to csv file.. ", end='', flush=True)
        self._save_to_csv(midi_commands, f"music/{save_id}.csv")
        print("Done.")

        print("Converting to MIDI.. ", end='', flush=True)
        self._convert_csv_file(f"music/{save_id}.csv", f"music/{save_id}.midi")
        print("Done.")
        print(f"Music file {save_id}.midi has been generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generates new music from a previously trained model")
    parser.add_argument("model_path", type=str, help="Compulsory argument. Path to a saved model.")
    parser.add_argument("--csvmidi_path", type=str, help="Path to Csvmidi.exe. If on Unix, may need to recompile", default="preprocessing/Csvmidi.exe")
    parser.add_argument("--start_path", type=str, help="Path to starting data. If None, random?", default=None)
    parser.add_argument("--start_length", type=int, help="Length of starter data, should be equal to length of training examples. (?)", default=200)
    parser.add_argument("--gen_len", type=int, help="Length of the song to be generated in number samples.", default=1000)
    args = parser.parse_args()

    CSVMIDI_PATH = args.csvmidi_path

    generator = MusicGenerator(args.model_path, args.csvmidi_path)
    generator.generate(start_data=args.start_path, start_data_len=args.start_length, gen_len=args.gen_len)
