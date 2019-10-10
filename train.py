import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

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
 
import pickle

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

# Loads token list and produces token_id-argmax dictionary and argmax-token dictionary
def load_token_list(path):
    f = open(path, 'rb')
    token_list = pickle.load(f)
    f.close()

    # First produce token_id-argmax
    token_dict = {}
    for t in token_list:
        token_dict[t[1]] = t[0]

    # Next produce argmax_token
    argmax_dict = {}
    for t in token_list:
        argmax_dict[t[0]] = t[2]

    return token_dict, argmax_dict 


params = {
    'meta_file': "./preprocessing/np_out/META.csv",
    'input_shape': (200, 176),
    'shuffle': True,
    'batch_size': 64,
    'alpha': 0.0001,
    'model_path': "./models",
    'epochs': 20,
    'nb_workers': 8,
    'val_split_percent': 0.005,
    'seq_len': 200,
    'nb_tokens': 1000,
    'vocab_size': 128
}

print("Generating ID List.. ", end='')
ID_list = multi_ID_list_generation(params['meta_file'], params['input_shape'][0], workers=params['nb_workers'])

random.shuffle(ID_list)
print("Done.")

print("Loading token dictionary.. ", end='')
token_dict, argmax_dict = load_token_list("./preprocessing/token_list.pkl")
print("Done.")

print("Creating Data Generators.. ", end='')
val_split = int(len(ID_list) * params['val_split_percent'])
training_generator = DataGenerator(params['nb_tokens'] + 1, ID_list[:-val_split], token_dict, shuffle=params['shuffle'], batch_size=params['batch_size'])
validation_generator = DataGenerator(params['nb_tokens'] + 1, ID_list[-val_split:], token_dict, shuffle=False, batch_size=params['batch_size'])
print("Done.")
print(f"{len(ID_list) - val_split} samples of training.\n{val_split} samples for validation.")

print("Creating Model.. ", end='')
model = create_model(params['seq_len'], params['nb_tokens'] + 1, params['vocab_size'])
print("Done.")

print("Creating Optimiser.. ", end='')
opt = Adam(lr=params['alpha'])
print("Done.")

print("Compiling Model.. ", end='')
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
print("Done.")

# Start training
start_time = int(time.time())
mdl_check = ModelCheckpoint(f"{params['model_path']}/tchAIkovsky-" + str(start_time) + "-{epoch:02d}.h5")
model.fit_generator(generator=training_generator, validation_data=validation_generator,
                    epochs=params['epochs'], steps_per_epoch=len(ID_list[:-1000]) // params['batch_size'],
                    use_multiprocessing=False, workers=params['nb_workers'],
                    callbacks=[mdl_check])