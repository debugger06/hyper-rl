#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import random
from CNN import train_and_score


# In[ ]:





# In[ ]:


class NeuralNetwork:

    def __init__(self, nn_param_choice):
        self.nn_param_choices = nn_param_choice
        self.accuracy = 0
        self.network = {}
        self.hyper_params = {}

    def create_random_network(self):
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        self.network = network

    def train(self, x_train, y_train, x_test, y_test):
        self.accuracy = train_and_score(self.network, x_train, y_train, x_test, y_test)

