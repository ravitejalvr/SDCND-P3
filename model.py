'''
Created on Jul 22, 2017
model.py is used for generating model.h5, which is used to create video file of autonomous vehicle.
@author: ravit
'''

import cv2
from keras.layers import Flatten, Dense,  Cropping2D, Conv2D, Lambda
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_shape = (66, 200, 3)
img_height, img_width, channels = input_shape

def image_augment(img_path, angle):
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    br_ratio = 1.0 + 0.3 * (np.random.rand() - 0.5)
    img[:, :, 2] = img[:, :, 2] * br_ratio
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # Flipping Image for two-side steering
    if np.random.randint(2) == 0:
        return img, angle
    return np.fliplr(img), -angle

def generator(log, batch_size=32, data_path='../data/IMG/'):
    
    # Correction for coming to center of the road
    correc = 0.3
    steer_correc = [0, correc, -correc]

    header = list(log.columns.values)

    images = []
    angles = []
    while True:
        idx = np.random.randint(len(log))

        image_select = np.random.randint(3)  

        image_path = data_path + log[header[image_select]].iloc[idx].split('/')[-1]
        angle = log['steering'].iloc[idx] + steer_correc[image_select]

        image, angle = image_augment(image_path, angle)
        images.append(image)
        angles.append(angle)

        if len(images) >= batch_size:
            yield shuffle(np.array(images), np.array(angles))
            images = []
            angles = []


model = Sequential()

model.add(Lambda(lambda x: (x/255) -0.5,input_shape = input_shape))

model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    # Second Convoluted layer with depth to 36
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    # Third convoluted layer with depth to 48
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    # Fourth Convoluted layer with depth to 64
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # Fifth Convoluted layer with depth to 64
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # Flatten Image
model.add(Flatten())
    # Reduce layers to 100
model.add(Dense(100,activation='relu'))
    # Reduce layers to 50
model.add(Dense(50,activation='relu'))
    # Reduce layers to 10
model.add(Dense(10,activation='relu'))    
    # Final Layer for steeering angle
model.add(Dense(1,activation='relu'))          
    # Selection of optimizer and error. Selected MSE as it is only one value

data_log = pd.read_csv('../data/driving_log.csv')
training_data, validation_data = train_test_split(data_log, test_size=0.2)

# Get batches using Generator Function
training_generator = generator(training_data, batch_size=512)
validation_generator = generator(validation_data, batch_size=512)

print('Training...\n')

model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4))
print(model.summary())

batch_size=512
history_object = model.fit_generator(training_generator, steps_per_epoch=len(training_data) // batch_size,
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_data) // batch_size,
                                         epochs=5,verbose = 1)
# Saved Model
model.save('model.h5')
print('Model Saved...:)')