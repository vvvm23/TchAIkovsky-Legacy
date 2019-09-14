from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten

# TODO: Parameterise this properly
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(Dense(176, activation='sigmoid'))
    return model
   
model = create_model((100, 176))
model.summary()