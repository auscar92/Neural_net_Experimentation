# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:52:34 2019

@author: ausca
"""


import pandas as p
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn import datasets


# =============================================================================
# #import the data
# names = ['White_King_col', 'White_King_row', 'White_Rook_col', 'White_Rook_row', 'Black_King_col', 'Black_King_row', 'depth_of_win']
# chess_data = p.read_csv("krkopt.csv", names = names)
# cleanup_nums = {"White_King_col": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7},
#                 "White_Rook_col": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7},
#                 "Black_King_col": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7},
#                 "depth_of_win": {"draw": 0, "zero": 1, "one": 2, "two": 3, "three": 4, "four": 5, "five": 6, "six": 7, "seven": 8, "eight": 9, 
#                                  "nine": 10, "ten": 11, "eleven": 12, "twelve": 13, "thirteen": 14, "fourteen": 15, "fifteen": 16, "sixteen": 17}}
# chess_data.replace(cleanup_nums, inplace=True)
# 
# #prepare the data
# X = chess_data.drop(columns=['depth_of_win']).values
# Y = chess_data["depth_of_win"].values.flatten()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# 
# #Setting up the model
# chess_model = tf.keras.Sequential([
#         #add a layer with 8 units
#         layers.Dense(8, activation='relu', input_shape=(6,)),
#         #add another
#         layers.Dense(16, activation='relu'),
#         #add a softmax layer with 18 output units
#         layers.Dense(18, activation='softmax')])
# 
# chess_model.compile(optimizer=tf.train.GradientDescentOptimizer(0.05),
#                loss='sparse_categorical_crossentropy',
#                metrics=['accuracy'])
# 
# #fit the model and evaluate the results
# chess_model.fit(X_train, Y_train, epochs = 50, batch_size=20)
# print('chess_model test batch results')
# chess_model.evaluate(X_test, Y_test, batch_size=20)
# =============================================================================

# =============================================================================
# Second Dataset
# =============================================================================
#import the data
names = ['Vendor_name', 'Model_name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
machine_data = p.read_csv("machine.csv", names = names)

machine_data = machine_data.drop(columns=['Model_name'])
machine_data.Vendor_name = machine_data.Vendor_name.astype('category')
machine_data["Vendor_name"]= machine_data.Vendor_name.cat.codes

#prepare the data
X = machine_data.drop(columns=['ERP']).values
Y = machine_data["ERP"].values.flatten()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Setting up the model
machine_model = tf.keras.Sequential([
        #add a layer with 8 units
        layers.Dense(8, activation='sigmoid', input_shape=(8,)),
        #add another
        layers.Dense(16, activation='sigmoid'),
        #add a softmax layer with 18 output units
        layers.Dense(8, activation='softmax')])

machine_model.compile(optimizer=tf.train.GradientDescentOptimizer(0.05),
                      loss='mse',
                      metrics=['mae'])

#fit the model and evaluate the results
machine_model.fit(X_train, Y_train, epochs = 50, batch_size=20)
print('machine_model test batch results')
machine_model.evaluate(X_test, Y_test, batch_size=20)

# =============================================================================
# Starting next dataset
# =============================================================================
cancer_data = datasets.load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Setting up the model
Cancer_model = tf.keras.Sequential([
        #add a layer with 8 units
        layers.Dense(32, activation='sigmoid', input_shape=(30,)),
        #add another
        layers.Dense(20, activation='sigmoid'),
        #add a softmax layer with 18 output units
        layers.Dense(8, activation='softmax')])

Cancer_model.compile(optimizer=tf.train.GradientDescentOptimizer(0.05),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

#fit the model and evaluate the results
Cancer_model.fit(X_train, Y_train, epochs = 50, batch_size=20)
print('Cancer_model test batch results')
Cancer_model.evaluate(X_test, Y_test, batch_size=20)

# =============================================================================
# Starting next dataset
# =============================================================================
names = ['letter', 'x_box', 'y_box', 'width', 'high', 'onpix', 'x_bar', 'y_bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x_ege', 'xegvy', 'y_ege', 'yegvx']
l_data = p.read_csv("letter-recognition.csv", names = names)

#this changes the letter column to numerical
l_data.letter.value_counts()
l_data.letter = l_data.letter.astype('category')
l_data["letter_cat"]= l_data.letter.cat.codes

#this gets rid of the old letter column
l_data = l_data.drop(columns=['letter'])

#prepares the x and y for seperation into train and test
X = l_data.drop(columns=["letter_cat"]).values
Y = l_data["letter_cat"].values.flatten()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Setting up the model
letter_model = tf.keras.Sequential([
        #add a layer with 8 units
        layers.Dense(32, activation='sigmoid', input_shape=(16,)),
        #add another
        layers.Dense(20, activation='sigmoid'),
        #add a softmax layer with 18 output units
        layers.Dense(26, activation='softmax')])

letter_model.compile(optimizer=tf.train.GradientDescentOptimizer(0.05),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

#fit the model and evaluate the results
letter_model.fit(X_train, Y_train, epochs = 50, batch_size=20)
print('letter_model test batch results')
letter_model.evaluate(X_test, Y_test, batch_size=20)