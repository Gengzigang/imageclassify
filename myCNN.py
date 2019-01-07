import numpy as np
from function import *
import dataset

class CNN:
    #alpha: learning rate
    def __init__(self,alpha):
        self.w1 = 0.2*(np.random.random((3,3,3,32))-0.5) # 3*3*3*32 shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters]
        self.b1 = 0.2*(np.random.random(32)-0.5)  # 32 num_filters
        #self.b1 = np.zeros(32)
        
        self.w2 = 0.2*(np.random.random((4,4,32,32))-0.5) #[-1, num_features] = [-1 , 30752]
        self.b2 = 0.2*(np.random.random(32)-0.5)

        self.w3 = 0.2*(np.random.random((3,3,32,64))-0.5) #[-1, num_features] = [-1 , 30752]
        self.b3 = 0.2*(np.random.random(64)-0.5)

        self.w4 = 0.2*(np.random.random((3,3,64,64))-0.5) #[-1, num_features] = [-1 , 30752]
        self.b4 = 0.2*(np.random.random(64)-0.5)

        self.w5 = 0.2*(np.random.random((256,64))-0.5) #[-1, num_features] = [-1 , 30752]
        self.b5 = 0.2*(np.random.random(64)-0.5)

        self.w6 = 0.2*(np.random.random((64,20))-0.5) #[-1, num_features] = [-1 , 30752]
        self.b6 = 0.2*(np.random.random(20)-0.5)

        self.w5 = self.w5.astype('float32')
        self.b5 = self.b5.astype('float32')

        self.w6 = self.w6.astype('float32')
        self.b6 = self.b6.astype('float32')

        self.alpha = alpha

    def load(self,num): 
        x = np.load('./b'+str(num)+'.npy', encoding="latin1")
        self.w1 = x[0]
        self.b1 = x[1]

        self.w2 = x[2]
        self.b2 = x[3]

        self.w3 = x[4]
        self.b3 = x[5]

        self.w4 = x[6]
        self.b4 = x[7]

        self.w5 = x[8]
        self.b5 = x[9]

        self.w6 = x[10]
        self.b6 = x[11]


    def forward(self,image,image_label):
        batch = image.shape[0]

        layer1_raw , col_X = conv_2D(image,self.w1,self.b1) #batch * 31 * 31 * 32
        layer1_pooling, index = maxpooling_forward(layer1_raw) #batch * 31 * 31 * 32 include relu

        layer1_raw , col_X = conv_2D(layer1_pooling,self.w2,self.b2) #batch * 31 * 31 * 32
        layer1_pooling, index = maxpooling_forward(layer1_raw) #batch * 31 * 31 * 32 include relu

        layer1_raw , col_X = conv_2D(layer1_pooling,self.w3,self.b3) #batch * 31 * 31 * 32
        layer1_pooling, index = maxpooling_forward(layer1_raw) #batch * 31 * 31 * 32 include relu

        layer1_raw , col_X = conv_2D(layer1_pooling,self.w4,self.b4) #batch * 31 * 31 * 32
        layer1_pooling, index = maxpooling_forward(layer1_raw) #batch * 31 * 31 * 32 include relu

        layer_flatten = layer1_pooling.reshape(( batch,-1)) # batch * 2304

        layer_fc = fc_forward(layer_flatten,self.w5,self.b5) # batch * 20

        layer_fc = relu_forward(layer_fc)

        layer_fc = fc_forward(layer_fc,self.w6,self.b6) # batch * 20

        h2 = softmax(layer_fc) # batch * 20
        return h2

    
    def backward(self,image,image_label):
        batch = image.shape[0]

        layer1_raw , col_X1 = conv_2D(image,self.w1,self.b1) #batch * 31 * 31 * 32
        layer1_pooling, index1 = maxpooling_forward(layer1_raw) #batch * 31 * 31 * 32 include relu

        layer2_raw , col_X2 = conv_2D(layer1_pooling,self.w2,self.b2) #batch * 31 * 31 * 32
        layer2_pooling, index2 = maxpooling_forward(layer2_raw) #batch * 31 * 31 * 32 include relu

        layer3_raw , col_X3 = conv_2D(layer2_pooling,self.w3,self.b3) #batch * 31 * 31 * 32
        layer3_pooling, index3 = maxpooling_forward(layer3_raw) #batch * 31 * 31 * 32 include relu

        layer4_raw , col_X4 = conv_2D(layer3_pooling,self.w4,self.b4) #batch * 31 * 31 * 32
        layer4_pooling, index4 = maxpooling_forward(layer4_raw) #batch * 31 * 31 * 32 include relu

        layer_flatten = layer4_pooling.reshape(( batch,-1)) # batch * 2304

        layer_fc5 = fc_forward(layer_flatten,self.w5,self.b5) # batch * 20

        layer_fc5 = relu_forward(layer_fc5)

        layer_fc6 = fc_forward(layer_fc5,self.w6,self.b6) # batch * 20

        h2 = softmax(layer_fc6) # batch * 20
        
        loss = np.sum(-image_label*np.log(h2)) /  batch
        
        delta6 = h2 - image_label # batch * 20

        grad_b6 = np.mean(delta6,axis=0) # 1 * 20
        
        grad_w6 = np.dot(layer_fc5.T,delta6) / batch

        delta5 = np.dot(delta6,self.w6.T)

        grad_b5 = np.mean(delta5,axis=0)

        grad_w5 = np.dot(layer_flatten.T,delta5) / batch

        grad_X5 = np.dot(delta5,self.w5.T).reshape( [batch,2,2,64] )

        grad_raw4 = maxpooling_backward(grad_X5,index4).reshape([-1,64])
        
        grad_w4 = (np.dot((col_X4.reshape([-1,576])).T,grad_raw4) / batch).reshape([3,3,64,64])

        grad_b4 = np.sum(grad_raw4,axis=0) / batch

        temp = np.pad(grad_raw4.reshape([batch,4,4,64]),((0,0),(2,2),(2,2),(0,0)),'constant')
        grad_X4,temp = conv_2D(temp,self.w4,np.zeros(64))

        grad_raw3 = maxpooling_backward(grad_X4,index3).reshape([-1,64])

        grad_w3 = (np.dot((col_X3.reshape([-1,288])).T,grad_raw3) / batch).reshape([3,3,32,64])

        grad_b3 = np.sum(grad_raw3,axis=0) / batch

        temp = np.pad(grad_raw3.reshape([batch,12,12,64]),((0,0),(2,2),(2,2),(0,0)),'constant')
        grad_X3 = conv_3D(temp,self.w3)

        grad_raw2 = maxpooling_backward(grad_X3,index2).reshape([-1,32])

        grad_w2 = (np.dot((col_X2.reshape([-1,512])).T,grad_raw2) / batch).reshape([4,4,32,32])

        grad_b2 = np.sum(grad_raw2,axis=0) / batch

        temp = np.pad(grad_raw2.reshape([batch,28,28,32]),((0,0),(3,3),(3,3),(0,0)),'constant')
        grad_X2 = conv_3D(temp,self.w2)
        
        grad_raw1 = maxpooling_backward(grad_X2,index1).reshape([-1,32])

        grad_w1 = (np.dot((col_X1.reshape([-1,27])).T,grad_raw1) / batch).reshape([3,3,3,32])

        grad_b1 = np.sum(grad_raw1,axis=0) / batch

        self.w1 = self.w1 - self.alpha * grad_w1
        self.b1 = self.b1 - self.alpha * grad_b1

        self.w2 = self.w2 - self.alpha * grad_w2
        self.b2 = self.b2 - self.alpha * grad_b2

        self.w3 = self.w3 - self.alpha * grad_w3
        self.b3 = self.b3 - self.alpha * grad_b3

        self.w4 = self.w4 - self.alpha * grad_w4
        self.b4 = self.b4 - self.alpha * grad_b4

        self.w5 = self.w5 - self.alpha * grad_w5
        self.b5 = self.b5 - self.alpha * grad_b5

        self.w6 = self.w6 - self.alpha * grad_w6
        self.b6 = self.b6 - self.alpha * grad_b6

        return loss
