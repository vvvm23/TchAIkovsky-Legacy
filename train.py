import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from data_generator import DataGenerator
from model import create_model

import pandas as pd
from tqdm import tqdm

import time

import random

# Function to generate ID_list from metafile
# TODO: Maybe multithread this using ThreadPoolExecutor. Right now this is annoyingly slow.
def ID_list_generation(meta_file, seq_size):
    df = pd.read_csv(meta_file, header=None, usecols=[0,1])
    fl_pairs = list(zip(df[0].tolist(), df[1].tolist()))
    _ID_list = []

    for pair in tqdm(fl_pairs):
        _ID_list = _ID_list + [(pair[0], i) for i in range(pair[1] - seq_size)]

    return _ID_list

print("Generating ID List.. ", end='')
ID_list = ID_list_generation("./preprocessing/np_out/META.csv", 200)
random.shuffle(ID_list)
print("Done.")

# TODO: Parameterisation, maybe in different file
params = {

}

print("Creating Model.. ", end='')
model = create_model((200, 176))
model.summary()
print("Done.")

# ID split is placeholder for now, potentially use maestro metadata to split?
print("Creating Data Generators.. ", end='')
training_generator = DataGenerator(176, ID_list[:-1000], shuffle=True, batch_size=64)
validation_generator = DataGenerator(176, ID_list[-1000:], shuffle=False, batch_size=64)
print("Done.")

# TODO: Adjust training parameters
print("Creating Optimiser.. ", end='')
opt = Adam(lr=0.0001) # Is this loss right? 
print("Done.")

print("Compiling Model.. ", end='')
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
print("Done.")

# Start training
start_time = int(time.time())
mdl_check = ModelCheckpoint("./models/tchAIkovsky-" + str(start_time) + "-{epoch:02d}.h5")
model.fit_generator(generator=training_generator, validation_data=validation_generator,
                    epochs=20, steps_per_epoch=len(ID_list[:-1000]) // 64,
                    use_multiprocessing=False, workers=8,
                    callbacks=[mdl_check])