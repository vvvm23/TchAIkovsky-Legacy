import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from data_generator import DataGenerator
from model import create_model

import pandas as pd
from tqdm import tqdm

import time

import random

import threading
import concurrent.futures
 
def multi_ID_list_generation(meta_file, seq_size, workers=8):
    worker_output = [[]] * 8

    def _worker_ID_list_generation(arg):
        fl_pairs = arg[0]
        worker_id = arg[1]
        for pair in fl_pairs:
            worker_output[worker_id] = worker_output[worker_id] + [(pair[0], i) for i in range(pair[1] - seq_size)]

    df = pd.read_csv(meta_file, header=None, usecols=[0,1])
    fl_pairs = list(zip(df[0].tolist(), df[1].tolist()))

    div = len(fl_pairs) // workers

    executor_args = [(fl_pairs[div*i:div*(i+1)], i) for i in range(workers - 1)]
    executor_args.append((fl_pairs[div*(workers - 1):], workers-1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(_worker_ID_list_generation, executor_args)

    return [x for y in worker_output for x in y]

print("Generating ID List.. ", end='')
#ID_list = ID_list_generation("./preprocessing/np_out/META.csv", 200)
ID_list = multi_ID_list_generation("./preprocessing/np_out/META.csv", 200, workers=8)

random.shuffle(ID_list)
print("Done.")

# TODO: Parameterisation, maybe in different file
params = {

}

print("Creating Model.. ", end='')
model = create_model((200, 176))
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