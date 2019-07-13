# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 20:55:58 2019

@author: barry
"""
import mnist_loader

import network

if __name__ == '__main__':
    
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()

    net = network.Network([784, 30, 10])

    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)        
    