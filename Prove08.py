# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:48:35 2019

@author: ausca
"""

import pandas as p
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

NAME1 = "Chess-Classifier-{}".format(int(time.time()))

tensorboard1 = TensorBoard(log_dir='logs/{}'.format(NAME1))

sess = tf.Session()
file_writer = tf.summary.FileWriter('logs', sess.graph)

#import the data
names = ['White_King_col', 'White_King_row', 'White_Rook_col', 'White_Rook_row', 'Black_King_col', 'Black_King_row', 'depth_of_win']
chess_data = p.read_csv("krkopt.csv", names = names)
cleanup_nums = {"White_King_col": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7},
                "White_Rook_col": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7},
                "Black_King_col": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7},
                "depth_of_win": {"draw": 0, "zero": 1, "one": 2, "two": 3, "three": 4, "four": 5, "five": 6, "six": 7, "seven": 8, "eight": 9, 
                                 "nine": 10, "ten": 11, "eleven": 12, "twelve": 13, "thirteen": 14, "fourteen": 15, "fifteen": 16, "sixteen": 17}}
chess_data.replace(cleanup_nums, inplace=True)

#prepare the data
X = chess_data.drop(columns=['depth_of_win']).values
Y = chess_data["depth_of_win"].values.flatten()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Setting up the model
chess_model = tf.keras.Sequential([
        #add a layer with 8 units
        layers.Dense(16, activation='relu', input_shape=(6,)),
        #add another
        layers.Dense(32, activation='relu'),
        #add another
        layers.Dense(48, activation='relu'),
        #add a softmax layer with 18 output units
        layers.Dense(18, activation='softmax')])

chess_model.compile(optimizer=tf.train.AdamOptimizer(0.002),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

#fit the model and evaluate the results
chess_model.fit(X_train, Y_train, epochs = 200, batch_size=40, callbacks=[tensorboard1])

# =============================================================================
# Second Dataset, regression
# =============================================================================

NAME2 = "Computer-part-Regression-{}".format(int(time.time()))

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

feature_columns=[tf.feature_column.numeric_column(key="Vendor_name"),
                 tf.feature_column.numeric_column(key="MYCT"), tf.feature_column.numeric_column(key="MMIN"), 
                 tf.feature_column.numeric_column(key="MMAX"), tf.feature_column.numeric_column(key="CACH"),
                 tf.feature_column.numeric_column(key="CHMIN"), tf.feature_column.numeric_column(key="CHMAX"),
                 tf.feature_column.numeric_column(key="PRP"), tf.feature_column.numeric_column(key="ERP")]

machine_model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)]
        )
optimizer = tf.keras.optimizers.RMSprop(0.002)
machine_model.compile(loss='mean_squared_error', 
                      optimizer=optimizer,
                      metrics=['mean_absolute_error'])

tensorboard2 = TensorBoard(log_dir='logs/{}'.format(NAME2))
machine_model.fit(X_train, Y_train, epochs = 200, batch_size=40, callbacks=[tensorboard2])