# Run this after making the training data from capturing the screen data
import os
import torch
import pickle
import cv2
import numpy as np
import random
from tqdm import tqdm
print('Imports complete')

# path to where the screen grab images are stored
path = f'F:\\Srujan\\MyProjects\\Data_Sets\\self_driving_car'

file_paths = []
for root, dirs, files in os.walk(path):
    for file in files:
        file_paths.append(f'{file}')

lefts = []
rights = []
forwards = []
backwards = []

for file in file_paths:
    if file.startswith('w'):
        forwards.append(file)
    elif file.startswith('s'):
        backwards.append(file)
    elif file.startswith('a'):
        lefts.append(file)
    elif file.startswith('d'):
        rights.append(file)
    else:
        pass

minima = min([len(forwards), len(backwards), len(lefts), len(rights)])
forwards = forwards[:minima]
backwards = backwards[:minima]
lefts = lefts[:minima]
rights = rights[:minima]
del(file_paths)
print(len(forwards), len(backwards), len(lefts), len(rights))

forward_label = torch.tensor([1, 0, 0, 0])
backward_label = torch.tensor([0, 1, 0, 0])
left_label = torch.tensor([0, 0, 1, 0])
right_label = torch.tensor([0, 0, 0, 1])

total_data = [forwards, backwards, lefts, rights]
labels = [forward_label, backward_label, left_label, right_label]
del(forwards, backwards, lefts, rights)

DATA = []

for i in range(len(total_data)):
     for data in tqdm(total_data[i]):
          img = cv2.imread(f'{path}\\{data}')
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          img = np.asarray(gray, dtype=np.float32)/255
          tensor_img = torch.tensor(img)
          DATA.append((tensor_img, labels[i]))
random.shuffle(DATA)
pickle_file = './self_driving_car.pickle'
with open(pickle_file, 'wb') as file:
     pickle.dump(DATA, file)
print('Data pickled')
