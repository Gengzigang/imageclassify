import numpy as np
import cv2
import dataset
import os
import time
import shelve

def conv_forward(X,W,b,stride=1,padding=1):
    '''
    input:X is image in H*W*C dimension, 
          W is the weight in WD*WH*WW*WC, 
          b is the bias in WD 
          D is the number of image, 
          C is the image channels,
          H*W is the size of image,
          WD is the num of filters,
          WC is the channels of filters is equal to the C,
          WH*WW is the size of filters
    output: the vector in H*W*WD
    '''
    shapex = X.shape
    shapeW = W.shape
    size = shapeW[2]*shapeW[3]*shapeW[1]
    x_pad = np.pad(X,(1,1),'constant',constant_values = (0,0))
    sh =(shapex[0],shapex[1],shapeW[0])
    out = np.zeros(shapeW[0]*shapex[1]*shapex[0]).reshape(*sh) #the size of the next layer
    for c in range(0,shapeW[0]):         #the number of filters 
        for i in range(0,shapex[0]):
            for j in range(0,shapex[1]): #input pixel
                temp = x_pad[i:i+shapeW[1],j:j+shapeW[2],1:shapex[2]+1] 
                out[i,j,c] = W[c].reshape(-1,size).dot(temp.reshape(size,-1))+b[c]
    cache = (x_pad,W,b)
    return out,cache

def conv_backward(cache,dout):
    '''input: cache:x_pad, dout:LOSS partial out
       output: dw:LOSS partial W, db:LOSS partial b, dx:LOSS partial x'''
    x_pad,W,b = cache
    shapex = x_pad.shape
    shapew = W.shape
    shapeo = dout.shape
    size = shapew[1]*shapew[2]*shapew[3]
    dw = np.zeros(size*shapew[0])
    sh = (shapew[0],shapew[1],shapew[2],shapew[3])
    shx = (shapex[0],shapex[1],shapex[2])
    dx = np.zeros(shapex[0]*shapex[1]*shapex[2]).reshape(*shx)
    dw = dw.reshape(*sh)
    db = np.zeros(shapew[0])
    for c in range(0,shapeo[2]):
        db[c] = np.sum(dout[:,:,c])
        for i in range(0,shapeo[0]):
            for j in range(0,shapeo[1]):
                dw[c] = dw[c]+dout[i,j,c]*x_pad[i:i+shapew[1],j:j+shapew[2],1:shapew[3]+1]
                dx[i:i+shapew[1],j:j+shapew[2],1:shapew[3]+1] += dout[i,j,c]*W[c]
    dx_real = dx[1:shapex[0]-1,1:shapex[1]-1,1:shapex[2]-1]
    return dw,db,dx_real
 
def maxpooling_forward(X,stride=2,padding=2):
    #X is the input, stride=2, padding=2
    #notice, it only used to particular size
    shapeX = X.shape
    sh = (shapeX[0]/2,shapeX[1]/2,shapeX[2])
    out = np.zeros(shapeX[0]*shapeX[1]*shapeX[2]/4).reshape(*sh)
    maxindex = []
    for c in range(0,shapeX[2]):
        for i in range(0,shapeX[0]/2):
            for j in range(0,shapeX[1]/2):
                temp = X[2*i:2*i+2,2*j:2*j+2,c]
                ind = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
                out[i,j,c] = temp[ind]
                maxindex.append([ind[0]+2*i,ind[1]+2*j,c])
    cache = (X,stride,padding,maxindex)
    return out,cache

def maxpooling_backward(dout,cache):
    X,stride,padding,maxindex = cache
    shapex = X.shape
    sh = (shapex[0],shapex[1],shapex[2])
    dx = np.zeros(shapex[0]*shapex[1]*shapex[2])
    dx = dx.reshape(*sh)
    while(maxindex != []):
        temp = maxindex.pop(0)
        dx[tuple(temp)] = dout[temp[0]/2,temp[1]/2,temp[2]]
    return dx

def relu_forward(X):
    out = np.maximum(X,0)
    return out

def relu_backward(dout,X):
    dx = dout.copy()
    dx[X < 0] = 0
    return dx

def fc_forward(X,W,b):
    out = X.T.dot(W)+b
    return out

def fc_backward(dout,x,w):
    shapex = x.shape
    shapew = w.shape
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = dout.copy()
    for i in range(0,shapex[0]):
        dx[i,0] = np.sum(dout*w[i,:])
        dw[i,:] = x[i,0]*dout
    return dx,dw,db

def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps)

def softmax_back(dout,sf):
    shapex = dout.shape
    I = np.diag(np.ones(shapex[1]))
    repetitions = (1, shapex[1])
    sj =  np.tile(sf,repetitions).reshape(shapex[1],shapex[1])
    temp = sj.T*(I-sj)
    dx = np.zeros(shapex[1])
    for i in range(0,shapex[1]):
        dx[i] = dout.dot(temp[:,i])
    return dx

def cross_entropy(pred,label):
    truelabel = np.argmax(label)
    predlabel = np.argmax(pred)
    if truelabel == predlabel:
        cache = 1
    else:
        cache = 0
    temp = -1*np.log(pred)
    loss = np.sum(temp*label)
    return loss,cache

def cross_entropy_back(pred,label):
    i = np.argmax(label)
    dp = np.zeros(pred.shape)
    dp[0][i] = -1/pred[0][i]
    return dp

classes = os.listdir('dataset')
num_classes = len(classes)

validation_size = 0.2
img_size = 64

train_path='dataset'

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:{}".format(len(data.train.labels)))
print("Number of files in Validation-set:{}".format(len(data.valid.labels)))

f = 0
learning_rate = 1e-3
loss = 10
w1 = 0.2*(np.random.random((32,3,3,3))-0.5)
b1 = 0.2*(np.random.random(32)-0.5)

w2 = 0.2*(np.random.random((32,3,3,32))-0.5)
b2 = 0.2*(np.random.random(32)-0.5)

w3 = 0.2*(np.random.random((64,3,3,32))-0.5)
b3 = 0.2*(np.random.random(64)-0.5)

w4 = 0.2*(np.random.random((4096,20))-0.5)
b4 = 0.2*(np.random.random(20)-0.5)
'''
filename = '/tmp/shelve.out'
my_shelf = shelve.open(filename,'n')

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        print('ERROR SHELVEING: {0}'.format(key))
my_shelf.close()'''
loss_sum = 0
count_size = 0
epoch = 0
true_size = 0
while(loss>0.00001):
    if f>=8000:
        f = 0
    image = data.train.images[f]
    image_label = data.train.labels[f]
    image_label = image_label.reshape((1,20))
    f = f+1
    count_size = count_size+1
    layer1_raw,cache1 = conv_forward(image,w1,b1)
    layer1_pooling,cachepool1 = maxpooling_forward(layer1_raw)
    layer1_relu = relu_forward(layer1_pooling)

    layer2_raw,cache2 = conv_forward(layer1_relu,w2,b2)
    layer2_pooling,cachepool2 = maxpooling_forward(layer2_raw)
    layer2_relu = relu_forward(layer2_pooling)

    layer3_raw,cache3 = conv_forward(layer2_relu,w3,b3)
    layer3_pooling,cachepool3 = maxpooling_forward(layer3_raw)
    layer3_relu = relu_forward(layer3_pooling)

    shapely3 = layer3_relu.shape
    layer3_flatten = layer3_relu.reshape(shapely3[0]*shapely3[1]*shapely3[2],-1)

    layer_fc = fc_forward(layer3_flatten,w4,b4)
    result = softmax(layer_fc)
    loss,whethertrue = cross_entropy(result,image_label)
    loss_sum = loss_sum+loss
    true_size = true_size+whethertrue
    dloss = cross_entropy_back(result,image_label)
    ds = softmax_back(dloss,result)
    dx4,dw4,db4 = fc_backward(ds,layer3_flatten,w4)
    w4 = w4-learning_rate*dw4
    b4 = b4-learning_rate*db4
    dx4 = dx4.reshape((shapely3[0],shapely3[1],shapely3[2]))
    dxrelu3 = relu_backward(dx4,layer3_pooling)
    dxpooling3 = maxpooling_backward(dxrelu3,cachepool3)
    dw3,db3,dxconv3 = conv_backward(cache3,dxpooling3)
    w3 = w3-learning_rate*dw3
    b3 = b3-learning_rate*db3

    dxrelu2 = relu_backward(dxconv3,layer2_pooling)
    dxpooling2 = maxpooling_backward(dxrelu2,cachepool2)
    dw2,db2,dxconv2 = conv_backward(cache2,dxpooling2)
    w2 = w2-learning_rate*dw2
    b2 = b2-learning_rate*db2

    dxrelu1 = relu_backward(dxconv2,layer1_pooling)
    dxpooling1 = maxpooling_backward(dxrelu1,cachepool1)
    dw1,db1,dxconv1 = conv_backward(cache1,dxpooling1)
    w1 = w1-learning_rate*dw1
    b1 = b1-learning_rate*db1

    if count_size == 32:
        count_size = 0
        loss_sum = loss_sum/32
        true_size = true_size/32
        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%},  Train Loss: {2:.3f}"
        print(msg.format(epoch + 1, true_size, loss_sum))
        epoch = epoch + 1
        true_size = 0
        loss_sum = 0










