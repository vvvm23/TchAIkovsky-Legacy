import pandas as pd
import os
from tqdm import tqdm
from music21 import converter, note, chord
from fractions import Fraction
import numpy as np

def round_down(num, factor):
    return num - (num % factor)

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


META_PATH = "./maestro-v2.0.0/maestro-v2.0.0.csv"
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

cwd = os.getcwd().replace('\\', '/')

notes = []

display_count = 0
for f in midi_list:
    display_count += 1
    print('\nProcessing ' + f + '. ' + str(display_count)  + '/' + str(len(midi_list)))
    
    print('Parsing file.. ', end='')
    stream = converter.parse(f)
    print('Done.')

    print('Flattening.. ', end='')
    elements = stream.flat.notes
    print('Done.')

    #current_time = 0
    #current_notes = []
    #note_stream = []
    note_dict = {}

    print('Ordering..')
    for element in tqdm(elements):
        current_time = round_down(float(Fraction(element.offset)), DUR_PRECISION)
        #if not element.offset == current_time:
        #    if element.offset < current_time:
        #        print("Offset in the past, ending..")
        #        exit()

            #note_stream.append((current_time, current_notes))

            #current_notes = []
            #current_time = float(Fraction(element.offset))

        if isinstance(element, note.Note):
            #current_notes.append(element)
            try:
                note_dict[current_time].append(element)
            except KeyError as e:
                note_dict[current_time] = [element]
        elif isinstance(element, chord.Chord):
            #current_notes = current_notes + list(element.notes)
            try:
                note_dict[current_time] = note_dict[current_time] + [n for n in element.notes]
            except KeyError as e:
                note_dict[current_time] = [n for n in element.notes]
    
    vector_seq = np.zeros((int(max(note_dict) / DUR_PRECISION) + 1, INPUT_SIZE))
    i = 0

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
        
    #nb_samples = vector_seq.shape[0] - SEQ_LEN
    #training_input = np.zeros((nb_samples, SEQ_LEN, INPUT_SIZE))
    #training_output = np.zeros((nb_samples, INPUT_SIZE))

    #for i in range(nb_samples):
    #    training_input[i, :, :] = vector_seq[i:i+SEQ_LEN, :].reshape((1,SEQ_LEN, INPUT_SIZE))
    #    training_output[i, :] = vector_seq[i+SEQ_LEN, :]

    #print(training_input.shape)
    #print(training_output.shape)
    #note_stream.sort()
