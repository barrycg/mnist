# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 20:55:58 2019

@author: barry
"""

'''
import mnist_loader
import network
if __name__ == '__main__':
    
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()

    net = network.Network([784, 30, 10])

    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)        
'''

'''
import mnist_loader    
import network2
if __name__ == '__main__':
    
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()

    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,
            evaluation_data=validation_data,
            monitor_evaluation_accuracy=True)        
'''    

'''
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

if __name__ == '__main__':
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    net = Network([FullyConnectedLayer(n_in=784, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, 
                validation_data, test_data)
'''

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

if __name__ == '__main__':
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                          filter_shape=(20, 1, 5, 5),
                          poolsize=(2,2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, 
                validation_data, test_data)
    
'''
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

if __name__ == '__main__':    
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    
    net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                          filter_shape=(20, 1, 5, 5),
                          poolsize=(2, 2)),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                          filter_shape=(40, 20, 5, 5),
                          poolsize=(2,2)),
            FullyConnectedLayer(n_in=40*4*4, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.03,
                    validation_data, test_data)
'''

'''
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU

if __name__ == '__main__':    
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    
    net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                          filter_shape=(20, 1, 5, 5),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                          filter_shape=(40, 20, 5, 5),
                          poolsize=(2,2),
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.03,
                    validation_data, test_data, lmbda=0.1)
'''            
   
        
   