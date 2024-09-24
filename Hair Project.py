# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 07:33:10 2024

@author: sh032
"""

import os 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.layers import Dense,Conv2D, MaxPooling2D, Activation, BatchNormalization, Flatten
from keras import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import shutil

os.chdir(r"C:\Users\sh032\hair")
os.makedirs("train/Curly_Hair")
os.makedirs("train/Straight_Hair")
os.makedirs("train/Wavy_Hair")

os.makedirs("test/Curly_Hair")
os.makedirs("test/Straight_Hair")
os.makedirs("test/Wavy_Hair")

for c in random.sample(os.listdir(r"C:\Users\sh032\hair\Curly Hair"),200):
                        chemin=os.path.join(r"C:\Users\sh032\hair\Curly Hair",c)
                        shutil.move(chemin,r"train/Curly_Hair")
for c in random.sample(os.listdir(r"C:\Users\sh032\hair\Straight Hair"),200):
                       chemin=os.path.join(r"C:\Users\sh032\hair\Straight Hair",c)
                       shutil.move(chemin,r"train/Straight_Hair")
for c in random.sample(os.listdir(r"C:\Users\sh032\hair\Wavy Hair"),200):
                       chemin=os.path.join(r"C:\Users\sh032\hair\Wavy Hair",c)
                       shutil.move(chemin,r"train/Wavy_Hair")

for c in random.sample(os.listdir(r"C:\Users\sh032\hair\Curly Hair"),100):
                       chemin=os.path.join(r"C:\Users\sh032\hair\Curly Hair",c)
                       shutil.move(chemin,r"test/Curly_Hair")
for c in random.sample(os.listdir(r"C:\Users\sh032\hair\Straight Hair"),100):
                       chemin=os.path.join(r"C:\Users\sh032\hair\Straight Hair",c)
                       shutil.move(chemin,r"test/Straight_Hair")
for c in random.sample(os.listdir(r"C:\Users\sh032\hair\Wavy Hair"),100):
                       chemin=os.path.join(r"C:\Users\sh032\hair\Wavy Hair",c)
                       shutil.move(chemin,r"test/Wavy_Hair")
os.chdir("../../")

train_path=r"C:\Users\sh032\hair\train"
test_path=r"C:\Users\sh032\hair\test"


train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=r"C:\Users\sh032\hair\train",target_size=(300,300),classes=["Curly_Hair","Straight_Hair","Wavy_Hair"],batch_size=10)

test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=r"C:\Users\sh032\hair\test",target_size=(300,300),classes=["Curly_Hair","Straight_Hair","Wavy_Hair"],batch_size=10)


imgs,labels=next(test_batches)

model=Sequential([
    keras.Input(shape=(300,300,3)),
    
    Conv2D(filters=32, kernel_size=(3,3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2), strides=2),
    
    Conv2D(filters=64, kernel_size=(3,3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(filters=128, kernel_size=(3,3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(units=3,activation="Softmax"),
    
    ])


model.summary()

model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(0.0001),
    metrics=["accuracy"]
    ) 

model.fit(train_batches,validation_data=test_batches,epochs=10,verbose=2)

proba=model.predict(test_batches)


predictions=[]
for prob in proba:
    predictions.append(np.argmax(prob))
    
predictions=np.array(predictions).reshape((len(predictions),1))
print(predictions)

image=test_batches[10][0][5]

img=np.expand_dims(image, axis=0)
print(np.argmax(model.predict(img)))

import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()

