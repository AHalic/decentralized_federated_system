import os
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPool2D,Flatten,Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
from joblib import load, dump
import base64
import dill
import tempfile
from io import BytesIO

def define_model(INPUT_SHAPE, NUM_CLASSES) -> Sequential:
    """
    Define the model architecture

    Parameters
    ------------
    INPUT_SHAPE: tuple
        Shape of the input data
    NUM_CLASSES: int
        Number of classes

    Returns
    ------------
    model: Sequential
        Model architecture
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=INPUT_SHAPE))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def ModelBase64Encoder(model_weights):
    """
    Encode the model weight to base64

    https://stackoverflow.com/questions/60567679/save-keras-model-weights-directly-to-bytes-memory
    """
    bytes_container = BytesIO()
    dill.dump(model_weights, bytes_container)
    bytes_container.seek(0)
    bytes_file = bytes_container.read()
    base64File = base64.b64encode(bytes_file)
    return base64File

def ModelBase64Decoder(model_bytes):
    """
    Decode the base64 encoded model weight

    https://stackoverflow.com/questions/60567679/save-keras-model-weights-directly-to-bytes-memory
    """
    loaded_binary = base64.b64decode(model_bytes)
    loaded_object = tempfile.TemporaryFile()
    loaded_object.write(loaded_binary)
    loaded_object.seek(0)
    ObjectFile = load(loaded_object)
    loaded_object.close()
    return ObjectFile