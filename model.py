from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, TimeDistributed, CuDNNLSTM, Embedding

def create_model():
    model = Sequential()

    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(Dropout(0.5))

    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(Dropout(0.5))

    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(Dropout(0.5))

    model.add(LSTM(256, return_sequences=False, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(313, activation='softmax'))
    return model
