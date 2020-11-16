#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten, Dropout, Conv2D

from keras.models import Sequential
from keras.callbacks import EarlyStopping


import tensorflow as tf


# In[ ]:




early_stopper = EarlyStopping(patience=5)


def create_model_model(network, input_shape, number_of_classes):
    

    #nb_layers = network.get('n_layers', 2)
    #nb_neurons = network.get('n_neurons', 10)
    #activation = network.get('activations', 'sigmoid')
    #optimizer = network.get('optimizers', 'adam')

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='sigmoid'))
    
    return model


def train_and_score(network, x_train, y_train, x_test, y_test):
    
    model = create_model_model(network, input_shape, number_of_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_valid, Y_valid))
    
    score = model.evaluate(x_test, y_test, verbose=0)

    return score  # 1 is accuracy. 0 is loss.

