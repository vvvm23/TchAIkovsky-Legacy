import pandas as pd
import os
from tqdm import tqdm
from music21 import converter, note, chord
from music21.midi import MidiFile
from fractions import Fraction

META_PATH = "./maestro-v2.0.0/maestro-v2.0.0.csv"
DATA_DIR = "maestro-v2.0.0/"

df = pd.read_csv(META_PATH)
midi_list = [DATA_DIR + x for x in list(df.loc[:, 'midi_filename'])]

cwd = os.getcwd().replace('\\', '/')

notes = []

for f in tqdm(midi_list):
    #name = f.split('/')[-1]
    #out = 'csv_out/' + name[:-4] + 'csv'
    #os.system(cwd + '/Midicsv.exe ' + cwd + '/' + f + ' ' + cwd + '/' +  out)
    midi = converter.parse(f)
    elements = midi.flat.notes
    pitches = []
    for element in elements:
        # Dissassemble chords
        # Get timestamp
        # If it's different, append old timestamp list to master list and set new timestamp

        current_notes = []
        if isinstance(element, note.Note):
            pitches.append(str(float(Fraction(element.offset))) + ':' + str(element.pitch.midi) + '|' + str(float(Fraction(element.quarterLength))))
        elif isinstance(element, chord.Chord):
            pitches.append(' '.join(str(n.pitch.midi) + '|' + str(float(Fraction(n.quarterLength))) for n in element.notes))
            pitches[-1] = str(float(Fraction(element.offset))) + ':' + pitches[-1]

        print('\n'.join(pitches))

    #notes.append(pitches)