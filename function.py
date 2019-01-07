import numpy as np

def im2col(X,filter_size=3): # X: 64*64*3  
    # X is a 4d tensor([batchsize, width ,height, channel])
    X_col = []
    for i in range(0, X.shape[0] - filter_size + 1):
            for j in range(0, X.shape[1] - filter_size + 1):
                col = X[i:i + filter_size, j:j + filter_size, :].reshape([-1])
                X_col.append(col)
    return np.array(X_col) # 3844*27 = (62*62)*(1*3*3*3)

def conv_2D(X,w,b):
    # X: 2(batch)*64*64*3
    # w 3*3*3*32
    # b 32
    # out batch * 62 * 62 * 32
    shapeX = X.shape
    shapeW = w.shape # 3*3*3*32
    col_w = w.reshape([-1, shapeW[3] ]) # 27*32

    out = np.zeros([ shapeX[0],shapeX[1]-shapeW[0]+1,shapeX[2]-shapeW[1]+1,shapeW[3] ])

    col_X = []
    for i in range(0, shapeX[0]):
        col_X_i = im2col(X[i],shapeW[0])
        out[i] = np.reshape( np.dot(col_X_i, col_w ) + b, (shapeX[1]-shapeW[0]+1,shapeX[2]-shapeW[1]+1,shapeW[3]) )
        col_X.append(col_X_i)
    return out,np.array(col_X)

'''
def maxpooling_forward(X):
    # X: 2(batch)*64*64*3
    #X is the input, stride=2
    #notice, it only used to particular size
    shapeX = X.shape
    out = np.zeros((shapeX[0],shapeX[1]//2,shapeX[2]//2,shapeX[3]))

    for c in range(0,shapeX[0]):
        for i in range(0,shapeX[1]//2):
            for j in range(0,shapeX[2]//2):
                for k in range(0,shapeX[3]):
                    #out[c][i][j][k] = np.max(X[c,2*i:2*i+2,2*j:2*j+2,k].reshape(-1))
                    out[c][i][j][k] = np.max(X[c,2*i:2*i+2,2*j:2*j+2,k].reshape(-1))
                    out[c][i][j][k] = np.max((0,out[c][i][j][k]))
    return out

'''
def maxpooling_forward(X): #batch * 31 * 31 * 32
    shapeX = X.shape
    out = np.zeros((shapeX[0],shapeX[1]//2,shapeX[2]//2,shapeX[3]))
    maxindex = np.zeros((shapeX[0],shapeX[1],shapeX[2],shapeX[3]))
    for c in range(0,shapeX[0]):
        for i in range(0,shapeX[1]//2):
            for j in range(0,shapeX[2]//2):
                for k in range(0,shapeX[3]):
                    temp = X[c,2*i:2*i+2,2*j:2*j+2,k]
                    ind = np.unravel_index(np.argmax(temp, axis=None), (2,2))
                    if temp[ind] > 0:
                        out[c,i,j,k] = temp[ind]
                        maxindex[c,ind[0]+2*i,ind[1]+2*j,k] = 1
                    else:
                        out[c,i,j,k] = 0
    return out,maxindex # batch * 31 * 31 * 32


def relu_forward(X):
    out = np.maximum(X,0)
    return out

def relu_backward(dout,X):
    dout[X < 0] = 0
    return dout

def fc_forward(X,W,b):
    out = X.dot(W)+b
    return out

def softmax(x):
    x = x - np.max(x)
    exps = np.exp(x)
    return (exps.T/np.sum(exps,axis=1)).T

def maxpooling_backward(grad_X,index):
    shapeX = grad_X.shape
    out = np.zeros(np.shape(index))

    for c in range(0,shapeX[0]):
        for i in range(0,shapeX[1]):
            for j in range(0,shapeX[2]):
                for k in range(0,shapeX[3]):
                    out[c,2*i:2*i+2,2*j:2*j+2,k] = index[c,2*i:2*i+2,2*j:2*j+2,k]*grad_X[c,i,j,k]
    return out

def conv_3D(X,w):#no padding, stride=1

    
    shapex = X.shape # batch*64*64*3
    shapeW = w.shape # batch*62*62*32
    w = w.reshape(shapeW[0],shapeW[1],shapeW[3],shapeW[2])
    shapeW = w.shape

    out = np.zeros( (shapex[0],shapex[1]-shapeW[0]+1,shapex[2]-shapeW[1]+1,shapeW[3] ))
    for c in range(0,shapex[0]):         #the number of filters
        for i in range(0,shapex[1]-shapeW[0]+1):
            for j in range(0,shapex[2]-shapeW[1]+1):
                for k in range(0,shapeW[3] ):
                    out[c][i][j][k] = out[c][i][j][k] + np.sum(np.sum(np.sum(w[:,:,:,k]*X[c,i:i+shapeW[0],j:j+shapeW[0],:])))
    return out # 3*3*3*32
