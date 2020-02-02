from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout, Flatten, TimeDistributed, CuDNNLSTM, Embedding
import tensorflow as tf

def create_model(seq_len):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(256, input_shape=(seq_len, 317), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.LSTM(256, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(317, activation='softmax'))
    return model
