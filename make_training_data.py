import time
import numpy as np
import cv2
from PIL.ImageGrab import grab
import random
from pynput.keyboard import Key, Listener

path = 'F:\\Srujan\\MyProjects\\Data_Sets\\self_driving_car\\'


def get_press(key):
    bef = time.time()
    img = np.asarray(grab())
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    width_, height_ = int(gray.shape[1]*0.25), int(gray.shape[0]*0.25)
    resized = cv2.resize(gray, (width_, height_))
    key = str(key).strip("'")
    al_1 = random.choice(alphabets)
    al_2 = random.choice(alphabets)
    num_1 = random.randint(0, 10000)
    num_2 = random.randint(0, 10000)
    if key in char_set:
        cv2.imwrite(f'{path}{key}_{al_1}_{al_2}_{num_1}_{num_2}.jpg', resized)
    else:
        pass
    aft = time.time()
    print(f'{key} Pressed, processing speed = {aft-bef}')


def get_release(key):
    pass


char_set = 'wasdWASD'
lower_ = 'abcdefghijklmnopqrstuvwxyz'  # for random file names
upper_ = lower_ + lower_.upper()
alphabets = [char for char in upper_]


time.sleep(15)  # Wait 15 seconds for the player to start driving the car
with Listener(on_press=get_press, on_release=get_release) as listener:
    listener.join()
