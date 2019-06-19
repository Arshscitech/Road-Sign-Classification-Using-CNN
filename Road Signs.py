#
# Author: Arsh
# Created On: 11 May, 2019 at 16:33:26
# Username: arsh_16
#

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import pickle
import pandas as pd
import random
import cv2
from PIL import Image
import requests


with open('train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('test.p', 'rb') as f:
    test_data = pickle.load(f)
X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

assert(X_train.shape[0]==y_train.shape[0])
assert(X_test.shape[0]==y_test.shape[0])
assert(X_val.shape[0]==y_val.shape[0])

data = pd.read_csv("signnames.csv")
cols, no_classes, no_samples = 5, 43, []

# fig, axs = plt.subplots(nrows=no_classes, ncols=cols, figsize = (5, 100))
# fig.tight_layout()
# for i in range(cols):
#     for j, row in data.iterrows():
#         x_selected  = X_train[y_train==j]
#         axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :, :], cmap=plt.get_cmap('gray'))
#         axs[j][i].axis('off')
#         if i==2:
#             axs[j][i].set_title(str(j)+"-" + row["SignName"])
#             no_samples.append(len(x_selected))
# plt.show()
# print(no_samples)
# plt.figure(figsize=(12, 4))
# plt.bar(range(no_classes), no_samples)
# plt.title("Distribution of train dataset")
# plt.xlabel("Class number")
# plt.ylabel("Number of images")
# plt.show()

# histogram equalization is used to standardized the lightning in an image. It increases contrast in the image, which also helps in feature extraction.

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert to gray scale

def equalize(img):
    return cv2.equalizeHist(img) # Works for grayscale images only

# img = X_train[1000]
# img = gray(img)
# plt.imshow(img, cmap = plt.get_cmap('gray'))
# plt.title("Gray")
# plt.show()
# img = equalize(img)
# plt.imshow(img, cmap=plt.get_cmap('gray'))
# plt.title("Equalized Image")
# plt.show()

def preprocess(img):
    return equalize(gray(img))/255 # Equalizing and normalizing

X_train = np.array(list(map(preprocess, X_train)))
X_test = np.array(list(map(preprocess, X_test)))
X_val = np.array(list(map(preprocess, X_val)))

# adding 1 dimension needed for convolution (1 channel present)
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)
X_val = X_val.reshape(X_val.shape[0], 32, 32, 1)
# one hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    width_shift_range= 0.1, # (shift the width. 10 indicates shift by 10 pixels. 0.1 indicates shift by 10% of image)
    height_shift_range= 0.1, # same like width shift, shifts the height.
    zoom_range= 0.2, # indicates 0.8 zoom out and 1.2 zoom in
    shear_range= 0.1, # indicates sheer angle is 0.1 degrees
    rotation_range= 10 # indicates rotation angle is 10 degrees
)
datagen.fit(X_train)
# batches = datagen.flow(X_train, y_train, batch_size=20) # it's a generator so only generates when called. Increases training time as it's generated during training but a lot memory efficient.
# X_batch, y_batch = next(batches) # get the next item in batches
# fig, axs = plt.subplots(1, 15, figsize = (20, 5))
# fig.tight_layout()
# for i in range(15):
#     axs[i].imshow(X_batch[i].reshape(32, 32))
#     axs[i].axis('off')
# plt.show()

def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def modified_model(): # Increases accuracy over the above leNet model by increasing the no. of filters, adding 2 more conv layers, and reducing lr 10 0.001.
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = modified_model()
print(model.summary())
# h = model.fit(X_train, y_train, validation_data=(X_val, y_val), shuffle=1, batch_size=400, epochs=10,  verbose=1)
h = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch=2000, epochs=10, validation_data=(X_val, y_val), shuffle=1)
# steps per epoch - Total images u want per epoch to be generated. Here 2000 steps indicate 2000*50 images
#
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'])
plt.savefig("Loss")
plt.show()
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.savefig("Accuracy")
plt.show()


score = model.evaluate(X_test, y_test, verbose=0)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# Testing of Image on internet pics
r= requests.get("https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg", stream = True)
img = Image.open(r.raw)
plt.imshow(img, cmap = plt.get_cmap('gray'))
plt.show()

img = cv2.resize(np.array(img), (32, 32))
img = preprocess(img)
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

img = img.reshape(1, 32, 32, 1)
print("Predicted sign:", model.predict_classes(img))