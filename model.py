from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, TimeDistributed, CuDNNLSTM, Embedding

'''
# TODO: Parameterise this properly
def create_model(input_shape):
    model = Sequential()
    model.add(CuDNNLSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(CuDNNLSTM(64, return_sequences=True))
    model.add(Dropout(0.5))

    #model.add(CuDNNLSTM(64, return_sequences=True))
    #model.add(Dropout(0.5))

    model.add(CuDNNLSTM(128, return_sequences=False))
    model.add(Dropout(0.5))

    #model.add(TimeDistributed(Dense(128, activation='relu')))

    #model.add(Flatten())
    model.add(Dense(176, activation='sigmoid'))

    return model
'''
# input.shape = (batch_size, sequence_length)
def create_model(seq_length, dict_size, vocab_size):
    model = Sequential()
    model.add(Embedding(dict_size, vocab_size, input_length=seq_length))

    model.add(CuDNNLSTM(64, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(CuDNNLSTM(64, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(CuDNNLSTM(64, return_sequences=False))
    model.add(Dropout(0.5))

    model.add(Dense(dict_size, activation='sigmoid')) 

    return model
