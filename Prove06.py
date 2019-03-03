# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:44:15 2019

@author: ausca
"""

import pandas as p
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

#import the data
names = ['Card1_Suit', 'Card1_Rank', 'Card2_Suit', 'Card2_Rank', 'Card3_Suit', 'Card3_Rank', 'Card4_Suit', 'Card4_Rank', 'Card5_Suit', 'Card5_Rank', 'Poker_Hand']
poker_data = p.read_csv("poker-hand-training-true.csv", names = names)

#prepares x and y
X = poker_data.drop(columns=['Poker_Hand']).values
Y = poker_data["Poker_Hand"].values.flatten()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model1 = tf.keras.Sequential([
        #add a layer with 64 units
        layers.Dense(64, activation='sigmoid', input_shape=(10,)),
        #add another
        layers.Dense(64, activation='sigmoid'),
        #add a softmax layer with 9 output units
        layers.Dense(10, activation='softmax')])

model1.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model1.fit(X_train, Y_train, epochs = 20, batch_size=100)
print('Model_1 test batch results')
model1.evaluate(X_test, Y_test, batch_size=100)

#This model has one more layer of 64 nodes
model2 = tf.keras.Sequential([
        #add a layer with 64 units
        layers.Dense(64, activation='sigmoid', input_shape=(10,)),
        #add another
        layers.Dense(64, activation='sigmoid'),
        #add another layer
        layers.Dense(64, activation='sigmoid'),
        #add a softmax layer with 9 output units
        layers.Dense(10, activation='softmax')])

model2.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model2.fit(X_train, Y_train, epochs = 20, batch_size=100)
print('Model_2 test batch results')
model2.evaluate(X_test, Y_test, batch_size=100)

#This model has a smaller batch size of 20
model3 = tf.keras.Sequential([
        #add a layer with 64 units
        layers.Dense(64, activation='sigmoid', input_shape=(10,)),
        #add another
        layers.Dense(64, activation='sigmoid'),
        #add a softmax layer with 9 output units
        layers.Dense(10, activation='softmax')])

model3.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model3.fit(X_train, Y_train, epochs = 20, batch_size=20)
print('Model_3 test batch results')
model3.evaluate(X_test, Y_test, batch_size=20)

#This model has a lot more epochs
model4 = tf.keras.Sequential([
        #add a layer with 64 units
        layers.Dense(64, activation='sigmoid', input_shape=(10,)),
        #add another
        layers.Dense(64, activation='sigmoid'),
        #add a softmax layer with 9 output units
        layers.Dense(10, activation='softmax')])

model4.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model4.fit(X_train, Y_train, epochs = 200, batch_size=100)
print('Model_4 test batch results')
model4.evaluate(X_test, Y_test, batch_size=20)

#This model has a different activation function: relu
model5 = tf.keras.Sequential([
        #add a layer with 64 units
        layers.Dense(64, activation='relu', input_shape=(10,)),
        #add another
        layers.Dense(64, activation='relu'),
        #add a softmax layer with 9 output units
        layers.Dense(10, activation='softmax')])

model5.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model5.fit(X_train, Y_train, epochs = 20, batch_size=100)
print('Model_ test batch results')
model5.evaluate(X_test, Y_test, batch_size=20)