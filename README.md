# Hair-Project
The purpose of my project was to create a model that recognizes the hair type of the person based on a picture.

In order to do that, I programmed a CNN and I preprocessed pictures with VGG16 preprocessing.
For this project, I used the "Three Hair Types" dataset that I found on Kaggle which contains a thousands of images for three different hair types: straight hair, curly hair and wavy hair. The creator of this dataset is vyom bhatia and he precised: "I scrapped all these images from google images using a chrome extension and sorted them out, image by image. I feel bad because I cannot give credit to the owners and Data Ethics is something I have to improve in as a person."
Here is the link of the dataset: https://www.kaggle.com/datasets/vyombhatia/the-three-hair-types/data

Here is the code that I used:

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

#Importation of pictures

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

#Preprocessing of pictures

train_path=r"C:\Users\sh032\hair\train"
test_path=r"C:\Users\sh032\hair\test"

train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=r"C:\Users\sh032\hair\train",target_size=(300,300),classes=["Curly_Hair","Straight_Hair","Wavy_Hair"],batch_size=10)

test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=r"C:\Users\sh032\hair\test",target_size=(300,300),classes=["Curly_Hair","Straight_Hair","Wavy_Hair"],batch_size=10)

imgs,labels=next(test_batches)

model=Sequential([
keras.Input(shape=(300,300,3)),

#The CNN model

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

#to make predictions

def prediction(picture_path):

  pic=Image.open(picture_path)
  pic=pic.resize(300,300)
  pic=np.array(pic)
  pic=np.expand_dims(picture,axis=0)
  return np.argmax(model.predict(pic))

