import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from utils import *
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, BatchNormalization, Add, Flatten, Concatenate, AveragePooling2D, GlobalMaxPooling2D, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import time
from glob import glob
from random import shuffle
import sys
import os
import random
import cv2
import math
import linecache
import string
import skimage
import imageio
from time import time
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn import svm


################# network architecture of G ################

def generator_simplified_api(inputs, is_train=True, reuse=False):
    image_size = 512
    k = 5
    # 128, 64, 32, 16
    #s2, s4, s8, s16, s32, s64 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16), int(image_size/32), int(image_size/64)
    s2, s4, s8, s16, s32, s64 ,s128,s256 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16), int(image_size/32), int(image_size/64),int(image_size/128), int(image_size/256)
    batch_size = 25
    gf_dim = 16 # Dimension of gen filters in first Conv2d layer. [64]

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim*64*s128*s128, W_init=w_init,
                act = tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s128, s128, gf_dim*64], name='g/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*32, (k, k), out_size=(s64, s64), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*16, (k, k), out_size=(s32, s32), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim*8, (k, k), out_size=(s16, s16), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, gf_dim*4, (k, k), out_size=(s8, s8), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h4/batch_norm')
                
        net_h5 = DeConv2d(net_h4, gf_dim*2, (k, k), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h5/decon2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h5/batch_norm')      

        net_h6 = DeConv2d(net_h5, gf_dim*1, (k, k), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h6/decon2d')
        net_h6 = BatchNormLayer(net_h6, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h6/batch_norm')
                
        net_h7 = DeConv2d(net_h6,1, (k, k), out_size=(image_size, image_size), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h7/decon2d')
        logits = net_h7.outputs
        net_h7.outputs = tf.nn.tanh(net_h7.outputs)
    return net_h7, logits


################# network architecture of D ################
################# the process of extracting feature  ################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def discriminator_simplified_api(inputs, is_train=True, reuse=False):    
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)   
    
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        
        net_in = InputLayer(inputs, name='d/in')           
        Conv2d1 = Conv2d(net_in, 96,( 11, 11),  (4, 4), act=lambda x: tl.act.lrelu(x, 0.2),
        padding='SAME', W_init=w_init, name='Conv2d1') #(55,55,96)      
        pool1 = MaxPool2d(Conv2d1, filter_size=(3, 3), strides=(2, 2), name='pool1')#(27,27,96)       
        
        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        Conv2d2 = Conv2d(pool1, 256,(5, 5),  (1, 1),act=lambda x: tl.act.lrelu(x, 0.2),
        padding='SAME', W_init=w_init, name='Conv2d2')     
        pool2 = MaxPool2d(Conv2d2,filter_size=(3, 3), strides=(2, 2), name='pool2')#（13,13,256）
        #norm2 = tf.nn.lrn(pool2, 2, 2e-05, 0.75, name='norm2')#（13,13,256）
        pool2_ = FlattenLayer(pool2, name='d/pool2/flatten')
               
        # 3rd Layer: Conv (w ReLu)
        Conv2d3 = Conv2d(pool2, 384,(3, 3),  (1, 1), act=lambda x: tl.act.lrelu(x, 0.2),
        padding='SAME', W_init=w_init,name='Conv2d3')#有padding（13,13,384）
        pool3 = MaxPool2d(Conv2d3,filter_size=(3, 3), strides=(2, 2), name='pool3')
        pool3_ = FlattenLayer(pool3, name='d/pool3/flatten')
        
        # 4th Layer: Conv (w ReLu) splitted into two groups
        Conv2d4 = Conv2d(Conv2d3,384, (3, 3),  (1, 1),act=lambda x: tl.act.lrelu(x, 0.2),
        padding='SAME', W_init=w_init, name='Conv2d4')#分为两组(13,13,192)和（13,13,192）
        pool4 = MaxPool2d(Conv2d4,filter_size=(3, 3), strides=(2, 2), name='pool4')
        pool4_ = FlattenLayer(pool4, name='d/pool4/flatten')
        
        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        Conv2d5 = Conv2d(Conv2d4,256, (3, 3),  (1, 1),act=lambda x: tl.act.lrelu(x, 0.2),
        padding='SAME', W_init=w_init, name='Conv2d5')#(13,13,128)和（13,13,128）
        pool5 = MaxPool2d(Conv2d5,filter_size=(3, 3), strides=(2, 2), name='pool5')#（6,6,256）
        pool5_ = FlattenLayer(pool5, name='d/pool5/flatten')
        
        #feature
        feature = ConcatLayer(layers = [pool5_,pool4_], name ='d/concat_layer1')
        net_h6 = DenseLayer(feature, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h6/lin_sigmoid')
        logits = net_h6.outputs
        net_h6.outputs = tf.nn.sigmoid(net_h6.outputs)

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])#变成一行
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)#（-1,4096）

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)#（4096）

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')#（-1,2）
   
    return net_h6, logits, feature.outputs


###### GBM classification using SVM #########

acc = []
nums = [29]

for num in nums:
    X_train=np.load('features/features%d_train.npy'%num)
    y_train=np.load('features/label%d_train.npy'%num)
    X_test=np.load('features/features%d_test.npy'%num)
    y_test=np.load('features/label%d_test.npy'%num)
    #print(y_train.shape)
    #y_train=y_train1.reshape(256,1)
    #print(y_train.shape)
    #y_test=y_test.reshape(256,2)
    
    print("Fitting the classifier to the training set")
    t0 = time()
    C = 10.0  # SVM regularization parameter
    clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    
    print("Predicting...")
    t0 = time()
    y_pred = clf.predict(X_test)
    
    print ("Accuracy: %.3f" %(accuracy_score(y_test, y_pred)))
    acc.append(accuracy_score(y_test, y_pred))
print (acc)




    
