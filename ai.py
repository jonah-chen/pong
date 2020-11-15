import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import categorical_crossentropy


def build_model(layers, units, input_units, c=1e-4):
    """Build simple feedfoward model taking input of shape (8,) of the x and y coords of the ball for 3 frames, and y coord of oppenent and user paddle for 1 frame. Return shape (3,) of the probability of executing actions ['up', 'down', 'do_nothing']
    """
    m = Sequential()
    m.add(Dense(input_units, input_shape=(8,), kernel_regularizer=l2(l=c)))
    m.add(BatchNormalization())
    m.add(ReLU())
    for _ in range(layers):
        m.add(Dense(units, kernel_regularizer=l2(l=c)))
        m.add(BatchNormalization())
        m.add(ReLU())

    m.add(Dense(3, activation='softmax', kernel_regularizer=l2(l=c)))

    return m

if __name__ == "__main__":
    model = build_model(2, 128, 64)
    checkpoint = ModelCheckpoint('checkpoint', save_best_only=True, monitor='loss')
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=categorical_crossentropy, metrics=['accuracy'])
    # model.fit()
    model.save('models/0')

