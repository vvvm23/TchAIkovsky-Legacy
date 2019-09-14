import keras
from keras.optimizers import Adam

from data_generator import DataGenerator
from model import create_model

# Function to generate ID_list from metafile
def ID_list_generation(meta_file):
    df = pd.read_csv(meta_file, header=None, usecols=[0,1])
    fl_pairs = list(zip(df[0].tolist(), df[1].tolist()))
    _ID_list = []

    for pair in fl_pairs:
        _ID_list = _ID_list + [(pair[0], i) for i in range(pair[1] - self.seq_size + 1)]

    return _ID_list

ID_list = ID_list_generation("./preprocessing/np_out/META.csv")

# TODO: Parameterisation, maybe in different file
params = {

}

model = create_model()

# ID split is placeholder for now, potentially use maestro metadata to split?
training_generator = DataGenerator(176, ID_list[:-1000])
validation_generator = DataGenerator(176, ID_list[-1000:])

# TODO: Adjust training parameters
opt = Adam() 

model.compile(opt)

# Start training
# TODO: Checkpoints while training
model.fit_generator(generator=training_generator, validation_data=validation_generator,
                    epochs=20,
                    use_multiprocessing=True, workers=4)