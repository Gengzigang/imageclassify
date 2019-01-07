import numpy as np
import dataset
import os
import time
from myCNN import CNN
np.random.seed(1)

num_classes = 20

validation_size = 0.2
img_size = 64
classes = 20

classes = os.listdir('dataset')
num_classes = len(classes)

validation_size = 0.2
img_size = 64
num_channels = 3
train_path='dataset'
beta = 0.01

data = dataset.read_train_sets(train_path, img_size, num_classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:{}".format(len(data.train.labels)))
print("Number of files in Validation-set:{}".format(len(data.valid.labels)))

learning_rate = 0.01
x1 = CNN(learning_rate)
x1.load(1)

x2 = CNN(learning_rate)
x2.load(2)

x3 = CNN(learning_rate)
x3.load(3)

x4 = CNN(learning_rate)
x4.load(4)

x5 = CNN(learning_rate)
x5.load(5)

sum_size = 0
count_size = 0

while sum_size < 2000:
    #f = np.random.randint(0,2000)

    image = data.valid.images[sum_size]
    image_label = data.valid.labels[sum_size]
    image_label = image_label.reshape((1,20))
    image = image.reshape((1,64,64,3))
    batch = image.shape[0]

    pred = np.zeros(20)
    h2 = x1.forward(image,image_label)
    pred[np.argmax(h2)] = pred[np.argmax(h2)] + 1
    h2 = x2.forward(image,image_label)
    pred[np.argmax(h2)] = pred[np.argmax(h2)] + 1
    h2 = x3.forward(image,image_label)
    pred[np.argmax(h2)] = pred[np.argmax(h2)] + 1
    h2 = x4.forward(image,image_label)
    pred[np.argmax(h2)] = pred[np.argmax(h2)] + 1
    h2 = x5.forward(image,image_label)
    pred[np.argmax(h2)] = pred[np.argmax(h2)] + 1

    pred1 = np.argmax(pred)
    pred[np.argmax(pred)] = 0
    pred2 = np.argmax(pred)

    label = np.argmax(image_label)
    print(h2[0,pred1])
    if pred1 == label:
        count_size += 1
    elif pred2 == label:
        count_size += 1
    sum_size += 1
    #print(count_size / sum_size)

print("===============")
print(count_size / sum_size)




















    #if(i == 8000):
    #    print(np.mean(loss))
    #    i = 0
    #f = np.random.choice(8000,16)
    #image = data.train.images[0:2] # batch * 64 * 64 * 3
    #image_label = data.train.labels[0:2] # batch * 3
    #grad_w1,grad_b1,grad_w2,grad_b2,loss[i] = x.calculate_grad(image,image_label)
    #x.backward(grad_w1,grad_b1,grad_w2,grad_b2)
    #x.forward(image,image_label)
    #i = i + 1
