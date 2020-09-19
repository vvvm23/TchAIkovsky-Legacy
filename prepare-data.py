import glob

import numpy as np
import pandas as pd

from tqdm import tqdm

def floorN(num, divisor):
    return num - (num%divisor)

def csv_to_list(path):
    f = open(path, mode='r')
    seq = []
    
    lines = f.readlines()
    c_time = 0
    c_vel = 0

    for ie, line in enumerate(lines):
        args = line.split(", ")
        track = int(args[0])
        time = int(args[1])
        r_type = args[2]
        other = args[3:]

        # ignore events that are not "Note_on_c". (EG: Pedalling controls)
        if not r_type == "Note_on_c": 
            continue
        
        if not track == 2:
            print("All valid tracks should be with id 2. Please investigate")

        note_val = int(other[1])
        velocity = int(other[2])

        # drop notes that are outside the defined range
        if note_val > NOTE_MAX or note_val < NOTE_MIN:
            continue 

        if velocity > VEL_MAX:
            velocity = VEL_MAX
        elif velocity < VEL_MIN:
            velocity = VEL_MIN

        # shift values down so minimum is 0
        note_val -= NOTE_MIN

        if velocity == 0: # stop note event
            q_time = floorN(time, TIME_INC)
            if not q_time == c_time: # advance time
                d_time = q_time - c_time
                c_time += d_time
            
                while d_time > TIME_MAX:
                    seq.append(TIME_INDEX + TIME_MAX // TIME_INC)
                    d_time -= TIME_MAX

                seq.append(TIME_INDEX + d_time // TIME_INC)

            seq.append(OFF_INDEX + note_val)

        else: # start note event
            q_time = floorN(time, TIME_INC)
            q_vel = floorN(velocity, VEL_INC)

            if not q_time == c_time: # advance time
                d_time = q_time - c_time
                c_time += d_time

                while d_time > TIME_MAX:
                    seq.append(TIME_INDEX + TIME_MAX // TIME_INC)
                    d_time -= TIME_MAX
                seq.append(TIME_INDEX + d_time // TIME_INC)

            if not q_vel == c_vel:
                seq.append(VEL_INDEX + (q_vel - VEL_MIN) // VEL_INC)
                c_vel = q_vel

            seq.append(ON_INDEX + note_val)
    seq.append(EOS_INDEX)
    f.close()
    return seq

def list_to_np(seq):
    return np.array(seq, dtype=np.int16)

if __name__ == '__main__':
    NOTE_MAX = 108
    NOTE_MIN = 21
    NB_NOTES = NOTE_MAX - NOTE_MIN + 1

    TIME_INC = 8 # ms
    TIME_MAX = 1000 # ms
    NB_TIME = TIME_MAX // TIME_INC

    VEL_MIN = 0
    VEL_MAX = 124
    VEL_INC = 4
    NB_VEL = (VEL_MAX - VEL_MIN) // VEL_INC + 1

    PAD_INDEX = 0
    SOS_INDEX = PAD_INDEX + 1
    EOS_INDEX = SOS_INDEX + 1
    ON_INDEX = EOS_INDEX + 1
    OFF_INDEX = ON_INDEX + NB_NOTES
    TIME_INDEX = OFF_INDEX + NB_NOTES
    VEL_INDEX = TIME_INDEX + NB_TIME
    
    print(VEL_INDEX + NB_VEL)

    csv_files = glob.glob("./csv_out/*.csv")

    pb = tqdm(total=len(csv_files))

    # use i to give unique output names
    # may not correspond to csv id, but it does not have to
    for i, path in enumerate(csv_files): 
        int_list = csv_to_list(path)
        int_npy = list_to_np(int_list)
        np.save(f"./np_out/{i}.npy", int_npy)
        pb.update(1)
