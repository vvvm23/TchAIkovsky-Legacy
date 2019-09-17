from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, TimeDistributed, CuDNNLSTM

# TODO: Parameterise this properly
def create_model(input_shape):
    model = Sequential()
    model.add(CuDNNLSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(CuDNNLSTM(64, return_sequences=True))
    model.add(Dropout(0.5))

    #model.add(CuDNNLSTM(64, return_sequences=True))
    #model.add(Dropout(0.5))

    model.add(CuDNNLSTM(64, return_sequences=False))
    model.add(Dropout(0.5))

    #model.add(TimeDistributed(Dense(128, activation='relu')))

    #model.add(Flatten())
    model.add(Dense(176, activation='sigmoid'))

    return model
