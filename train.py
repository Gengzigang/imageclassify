import numpy as np
import cv2
import dataset
import os
import time
import shelve
from myCNN import CNN

num_classes = 20

validation_size = 0.2
img_size = 64

train_path='dataset'

data = dataset.read_train_sets(train_path, img_size, num_classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:{}".format(len(data.train.labels)))
print("Number of files in Validation-set:{}".format(len(data.valid.labels)))

f = 0
learning_rate = 0.01
loss = 10

x = CNN(learning_rate)
loss = np.zeros(8000)
i = 0

while(1):
    if(i == 8000):
        print(np.mean(loss))
        i = 0
    f = np.random.choice(8000,16)
    image = data.train.images[f] # batch * 64 * 64 * 3
    image_label = data.train.labels[f] # batch * 3

    loss = x.backward(image,image_label)
    print(loss)

    i = i + 1