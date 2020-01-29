import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

'''
    Preprocessing
        1) Call Midicsv on file.
        2) Read in resulting csv line by line
            - If new velocity
                * add velocity change event

            - If at old time
                * add corresponding event to int sequence
            - If at new time
                * add time change event to int sequence
                * add corresponding event to int sequence

        3) Take int sequence and one hot encode in vector sequence
        4) Save to .npy file (or .h5)
            
    Arguments
        max_note: Maximum midi note value (default 108)
        min_note: Minimum midi note value (default 21)
        max_time: Maximum time jump in ms (default 1000ms)
        time_interval: Interval between discrete time steps (default 8ms)
        max_vel: Maximum velocity
        min_vel: Minimum velocity
        vel_jump: Jump in velocity
        meta_data_path: path to maestro meta data file (default ./maestro-v2.0.0/maestro-v2.0.0.csv)
        output_dir: path to output directory (default ./np_out)
        midicsv_executable_path: path to midicsv.exe (default ./midicsv_executable_path)
'''
def update_npy_meta(f, name, length, split):
    f.write(f"{name}.npy, {length}, {split}\n")

def save_to_npy(seq, name, split):
    if split == "train":
        np.save(f"{OUTPUT_DIR}train/{name}.npy", seq)
    elif split == "validation":
        np.save(f"{OUTPUT_DIR}validation/{name}.npy", seq)

def seq_to_np(seq):
    n = len(seq)
    out = np.zeros((n, 2*NB_NOTES+NB_TIME+NB_VEL))
    out[np.arange(n), seq] = 1.0
    return out

def csv_to_seq(csv_path):
    #In the maestro dataset, there is only ever one track. So we can ignore track 1 and 2 completely

    seq = []
    
    f = open(csv_path, mode='r')
    lines = f.readlines()
    for line in lines:
        args = line.split(", ")
        track = args[0]
        time = args[1]
        r_type = args[2]
        other = args[3:]

    f.close()

def midi_to_csv(path, save_name):
    CSV_OUT = "./csv_out/"
    os.system(f"{MIDICSV_PATH} {path} {save_name}")    

def read_meta():
    df = pd.read_csv(META_DATA_PATH, usecols=["midi_filename", "split"])
    return df

def preprocess():
    npy_meta_file = open(f"{OUTPUT_DIR}/META.csv", mode='a')
    midi_dataframe = read_meta()

    for index, row in tqdm(midi_dataframe.iterrows()):
        csv_path = f"{CSV_OUT}{index}.csv"
        midi_to_csv(f"./maestro-v2.0.0/{row[1]}", csv_path)
        int_seq = csv_to_seq(csv_path)
        npy_seq = seq_to_np(int_seq)
        save_to_npy(npy_seq, index, row[0])
        update_npy_meta(npy_meta_file, index, npy_seq.shape[0], row[0])

    npy_meta_file.close()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser("Preprocess MIDI data")
    parser.add_argument("--max_note", type=int, help="Define maximum midi value to accept", default=108)
    parser.add_argument("--min_note", type=int, help="Define minimum midi value to accept", default=21)
    parser.add_argument("--max_time", type=int, help="Maximum time jump for one event in milliseconds", default=1000)
    parser.add_argument("--time_interval", type=int, help="Interval between discrete time steps", default=8)
    parser.add_argument("--max_vel", type=int, help="Maximum note velocity (volume)")
    parser.add_argument("--min_vel", type=int, help="Minimum note velocity (volume)")
    parser.add_argument("--vel_interval", type=int, help="Jump in discrete velocity levels")
    parser.add_argument("--meta_data_path", type=str, help="Path to maestro meta csv file", default="./maestro-v2.0.0/maestro-v2.0.0.csv")
    parser.add_argument("--output_dir", type=str, help="Path to output directory to place .npy files", default="./np_out/")
    parser.add_argument("--midicsv_executable_path", type=str, help="Path to midicsv.exe", default="Midicsv.exe")
    args = parser.parse_args()

    MAX_NOTE = args.max_note
    MIN_NOTE = args.min_note
    MAX_TIME = args.max_time
    TIME_INTERVAL = args.time_interval
    MAX_VEL = args.max_vel
    MIN_VEL = args.min_vel
    VEL_INTERVAL = args.vel_interval
    META_DATA_PATH = args.meta_data_path
    OUTPUT_DIR = args.output_dir
    MIDICSV_PATH = args.midicsv_executable_path

    CSV_OUT = "./csv_out/"
    NB_NOTES = MAX_NOTE - MIN_NOTE + 1
    NB_TIME = MAX_TIME // TIME_INTERVAL
    NB_VEL = (MAX_VEL - MIN_VEL) // VEL_INTERVAL

    preprocess()
    