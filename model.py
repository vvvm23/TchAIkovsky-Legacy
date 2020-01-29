from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout, Flatten, TimeDistributed, CuDNNLSTM, Embedding

def create_model(seq_len):
    model = Sequential()

    model.add(LSTM(128, input_shape=(seq_len, 317), return_sequences=True, activation='relu'))
    model.add(Dropout(0.5))

    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.5))

    model.add(LSTM(128, return_sequences=False, activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(317, activation='softmax'))
    return model
