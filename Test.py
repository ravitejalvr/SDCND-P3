'''
Created on Jul 22, 2017

@author: ravit
'''

import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import numpy as np
import cv2
log = pd.read_csv('../data/driving_log.csv')
img_path = '../data/IMG/' + log.iloc[5,:]['center'].split('/')[-1]
img = cv2.imread(img_path)
pyplot.imsave('normal.jpg',img)

img = cv2.imread(img_path)
img = cv2.resize(img, (200, 66), cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
br_ratio = 1.0 + 0.3 * (np.random.rand() - 0.5)
img[:, :, 2] = img[:, :, 2] * br_ratio
img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

pyplot.imsave('augmented.jpg',img)