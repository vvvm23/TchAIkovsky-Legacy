import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from data_generator import DataGenerator
from model import create_model

import pandas as pd
from tqdm import tqdm

import time

# Function to generate ID_list from metafile
def ID_list_generation(meta_file, seq_size):
    df = pd.read_csv(meta_file, header=None, usecols=[0,1])
    fl_pairs = list(zip(df[0].tolist(), df[1].tolist()))
    _ID_list = []

    for pair in tqdm(fl_pairs):
        _ID_list = _ID_list + [(pair[0], i) for i in range(pair[1] - seq_size)]

    return _ID_list

print("Generating ID List.. ", end='')
ID_list = ID_list_generation("./preprocessing/np_out/META.csv", 100)
print("Done.")

# TODO: Parameterisation, maybe in different file
params = {

}

print("Creating Model.. ", end='')
model = create_model((100, 176))
model.summary()
print("Done.")

# ID split is placeholder for now, potentially use maestro metadata to split?
print("Creating Data Generators.. ", end='')
training_generator = DataGenerator(176, ID_list[:-1000], shuffle=False)
validation_generator = DataGenerator(176, ID_list[-1000:])
print("Done.")

# TODO: Adjust training parameters
print("Creating Optimiser.. ", end='')
opt = Adam() # Is this loss right? 
print("Done.")

print("Compiling Model.. ", end='')
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
print("Done.")

# Start training
start_time = int(time.time())
mdl_check = ModelCheckpoint('./models/tchAIkovsky-{start_time}-{epoch:02d}.h5')
model.fit_generator(generator=training_generator, validation_data=validation_generator,
                    epochs=20, steps_per_epoch=len(ID_list[:-1000]) // 32,
                    use_multiprocessing=False, workers=1,
                    callbacks=[mdl_check])