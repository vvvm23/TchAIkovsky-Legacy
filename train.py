import os

#import keras

#from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint

#from data_generator import DataGenerator
#from model import create_model

import pandas as pd
from tqdm import tqdm

import time

import random

import threading
import concurrent.futures
 
import pickle

def multi_ID_list_generation(meta_file, seq_size, workers=8):
    worker_output_training = [[] for _ in range(workers)]
    worker_output_validation = [[] for _ in range(workers)]

    def _worker_ID_list_generation(arg):
        fl_pairs = arg[0]
        worker_id = arg[1]
        for pair in fl_pairs:
            if pair[2] == "train":
                worker_output_training[worker_id] = worker_output_training[worker_id] + [(pair[0], i) for i in range(0, pair[1] - seq_size, params['id_interval'])]
            if pair[2] == "validation":
                worker_output_validation[worker_id] = worker_output_validation[worker_id] + [(pair[0], i) for i in range(0, pair[1] - seq_size, params['id_interval'])]
            
            pbar.update(1)

    df = pd.read_csv(meta_file, header=None)
    fl_pairs = list(zip(df[0].tolist(), df[1].tolist(), df[2].tolist()))

    div = len(fl_pairs) // workers

    executor_args = [(fl_pairs[div*i:div*(i+1)], i) for i in range(workers - 1)]
    executor_args.append((fl_pairs[div*(workers - 1):], workers-1))

    with tqdm(total=len(fl_pairs)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(_worker_ID_list_generation, executor_args)

    return [x for y in worker_output_training for x in y], [x for y in worker_output_validation for x in y]

params = {
    'meta_file': "./preprocessing/np_out/META.csv",
    'input_shape': (200, 317), #?
    'shuffle': True,
    'batch_size': 64,
    'alpha': 0.001,
    'model_path': "./models",
    'epochs': 20,
    'nb_workers': 16,
    'seq_len': 200,
    'id_interval': 32
}

print("Generating ID List.. \n", end='', flush=True)
training_ID_list, validation_ID_list = multi_ID_list_generation(params['meta_file'], params['input_shape'][0], workers=params['nb_workers'])
random.shuffle(training_ID_list)
random.shuffle(validation_ID_list)
print("Done.")

'''
print("Creating Data Generators.. ", end='')
val_split = int(len(ID_list) * params['val_split_percent'])
training_generator = DataGenerator(params['nb_tokens'] + 1, training_ID_list, token_dict, shuffle=params['shuffle'], batch_size=params['batch_size'])
validation_generator = DataGenerator(params['nb_tokens'] + 1, validation_ID_list, token_dict, shuffle=False, batch_size=params['batch_size'])
print("Done.")
print(f"{len(training_ID_list)} samples of training.\n{validation_ID_list} samples for validation.")

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
'''