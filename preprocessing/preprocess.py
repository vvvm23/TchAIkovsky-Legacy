import pandas as pd
import os
from tqdm import tqdm
from music21 import converter, note, chord, midi
from fractions import Fraction
import numpy as np

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
DUR_PRECISION = 0.1
DUR_NORM = 4.0

df = pd.read_csv(META_PATH)
midi_list = [DATA_DIR + x for x in list(df.loc[:, 'midi_filename'])]

meta_f = open(NP_META_PATH, 'w')
display_count = 0
for save_id, f in enumerate(midi_list):
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
            try:
                note_dict[current_time].append(element)
            except KeyError as e:
                note_dict[current_time] = [element]
        elif isinstance(element, chord.Chord):
            try:
                note_dict[current_time] = note_dict[current_time] + [n for n in element.notes]
            except KeyError as e:
                note_dict[current_time] = [n for n in element.notes]
    
    vector_seq = np.zeros((int(max(note_dict) / DUR_PRECISION) + 1, INPUT_SIZE))
    i = 0

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

meta_f.close()