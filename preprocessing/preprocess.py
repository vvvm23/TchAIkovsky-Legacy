import pandas as pd
import os
from tqdm import tqdm
from music21 import converter, note, chord, midi
from fractions import Fraction
import numpy as np
import pickle

'''
    Maybe a better idea would be to have different input channels. One is note value, one is duration, one is on or off.
    Set number of channels, so input size is nb_channels*3
    Output same.
    Ignore if number of channels too large.
'''

def round_down(num, factor):
    return num - (num % factor)

# TODO: Replace with np.linspace
def frange(start, stop, jump, end=False, via_str=False):
    if via_str:
        start = str(start)
        stop = str(stop)
        jump = str(jump)
    start = Fraction(start)
    stop = Fraction(stop)
    jump = Fraction(jump)
    while start < stop:
        yield float(start)
        start += jump
    if end and start == stop:
        yield(float(start))

# TODO: Properly parameterise this
META_PATH = "./maestro-v2.0.0/maestro-v2.0.0.csv"
NP_META_PATH = "./np_out/META.csv"
DATA_DIR = "maestro-v2.0.0/"

MIN_MIDI = 21
MAX_MIDI = 108
NB_NOTES = MAX_MIDI - MIN_MIDI + 1
INPUT_SIZE = 2 * NB_NOTES
SEQ_LEN = 100
DUR_PRECISION = 0.25
DUR_NORM = 4.0
MAX_NOTES = 5

df = pd.read_csv(META_PATH)
midi_list = [DATA_DIR + x for x in list(df.loc[:, 'midi_filename'])]

meta_f = open(NP_META_PATH, 'w')
display_count = 0

nb_tokens = 1
token_dict = {"0-0-0-0-0": (0,0)} # CHANGE THIS

for save_id, f in enumerate(midi_list[:50]):
    display_count += 1
    print('\nProcessing ' + f + ' ' + str(display_count)  + '/' + str(len(midi_list)))
    
    print('Parsing file.. ', end='')
    stream = converter.parseFile(f, format='midi')
    print('Done.')

    print('Flattening.. ', end='')
    elements = stream.flat.notes
    print('Done.')

    note_dict = {}

    print('Ordering..')
    for element in elements: # This enumerate function is pretty good, use to improve older projects
        current_time = round_down(float(Fraction(element.offset)), DUR_PRECISION) # Lose some precision in time

        # Add to timestamp if exists, else create a new timestamp
        if isinstance(element, note.Note):
            if current_time in note_dict:
                note_dict[current_time].append(element)
            else:
                note_dict[current_time] = [element]
        elif isinstance(element, chord.Chord):
            if current_time in note_dict:
                note_dict[current_time] = note_dict[current_time] + [n for n in element.notes]
            else:
                note_dict[current_time] = [n for n in element.notes]
    
    token_seq = ["0-" * MAX_NOTES] * (int(max(note_dict) / DUR_PRECISION) + 1)
    token_seq = token_seq[:-1] 
    int_seq = np.zeros(int(max(note_dict) / DUR_PRECISION) + 1)

    vector_seq = np.zeros((int(max(note_dict) / DUR_PRECISION) + 1, INPUT_SIZE))
    i = 0

    for i, t in enumerate(frange(0.0, max(note_dict), DUR_PRECISION)):
        if not t in note_dict:
            continue
        
        #note_dict[t].sort(key=lambda x: x.pitch.midi)
        
        
        if len(note_dict[t]) <= MAX_NOTES:
            notes = sorted([x.pitch.midi for x in note_dict[t]])
            token_seq[i] = '-'.join(str(n) for n in notes)
        else:
            ran_sample = np.random.choice(note_dict[t], MAX_NOTES, replace=False)
            notes = sorted([x.pitch.midi for x in ran_sample])
            token_seq[i] = '-'.join(str(n) for n in notes)

        if not token_seq[i] in token_dict:
            token_dict[token_seq[i]] = (nb_tokens, 1)
            nb_tokens += 1
        else:
            token_dict[token_seq[i]] = (token_dict[token_seq[i]][0], token_dict[token_seq[i]][1] + 1)

        int_seq[i] = token_dict[token_seq[i]][0]

    '''
    # Iterate through timestamps based on DUR_PRECISION and vectorise
    for t in frange(0.0, max(note_dict), DUR_PRECISION):
        try:
            for n in note_dict[t]:
                vector_seq[i, (n.pitch.midi - MIN_MIDI)*2] = 1.0
                vector_seq[i, (n.pitch.midi - MIN_MIDI)*2 + 1] = n.quarterLength /  DUR_NORM
        except KeyError as e:
            pass
        except:
            raise

        i += 1

    # Save to file and update metafile
    np.save("./np_out/{0}.npy".format(save_id), vector_seq)
    meta_f.write("preprocessing/np_out/{0}.npy, {1}\n".format(save_id, vector_seq.shape[0]))
    '''
    np.save("./np_out/int_{0}.npy".format(save_id), int_seq)
    meta_f.write("preprocessing/np_out/int_{0}.npy, {1}\n".format(save_id, int_seq.shape[0]))
    meta_f.flush()

# Need to map real token IDs to top X ids for argmax output
TOP_TOKENS = 1000
sorted_tokens = sorted(token_dict.items(), key=lambda x: x[1][1], reverse=True)[:TOP_TOKENS]

save_list = [(0, 0, "0-0-0-0-0")] # change this magic string
for i, t in enumerate(sorted_tokens):
    save_list.append((i+1, t[1][0], t[0])) # This method means we have to convert from real token to argmax in DataGenerator
                                           # Test how slow this is.

print("\n".join(f"{t[0]} - {t[1]} - {t[2]}" for t in save_list))

with open('./token_list.pkl', 'wb') as f:
    pickle.dump(save_list, f, pickle.HIGHEST_PROTOCOL)

meta_f.close()