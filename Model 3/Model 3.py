# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:00:29 2021

@author: bhatn
"""
import keras
#import tensorflow as tf
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.applications.resnet import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
import numpy as np
import os
from sklearn.decomposition import PCA
from keras import backend as K
from tensorflow.core.protobuf import rewriter_config_pb2
from keras import initializers
from utils import *
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Dropout
from keras.layers import SeparableConv2D, Add, Convolution2D, concatenate, Layer, ReLU, DepthwiseConv2D, Reshape, Multiply, InputSpec
from keras.models import Model, load_model, model_from_json
from keras.regularizers import l2
from ftt import att
import matplotlib.pyplot as plt

class X_plus_Layer(Layer):
    def __init__(self, **kwargs):
        super(X_plus_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', initializer='zeros', trainable=True)
        super(X_plus_Layer, self).build(input_shape)

    def call(self, inpt_x):
        ''' block-level temporal attention '''
        x, A = inpt_x
        x_diag = x
        for i in range(25-1):
            x_diag = K.concatenate([x_diag, x], axis=2)
        x_diag_channals = x_diag
        x_diag_channals = K.expand_dims(x_diag, axis=3)
        x_diag = K.expand_dims(x_diag, axis=3)
        for i in range(3-1):
            x_diag_channals = K.concatenate([x_diag_channals, x_diag], axis=3)

        x_mask = x
        width = 25
        for w in range(width-1):
            x_mask = K.concatenate([x_mask, x], axis=2)
        x_mask_channals = x_mask
        x_mask_channals = K.expand_dims(x_mask, axis=3)
        x_mask = K.expand_dims(x_mask, axis=3)
        for i in range(3-1):
            x_mask_channals = K.concatenate([x_mask_channals, x_mask], axis=3)

        a_part = multiply([x_diag, A])
        a_part = self.alpha * a_part

        b_part = self.beta * x_mask_channals

        ans = Add()([a_part, b_part])
        return ans

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 300, 25, 3)
    # def get_config

def attention_map(inpt):
    s_o = inpt[0]
    t_o = inpt[1]
    ipt = inpt[2]
    channels=int(inpt[3])
    height = 300
    width = 25

    ''' adaptive spatial attention '''
    s_o = K.l2_normalize(s_o, axis=1)
    s_map = K.expand_dims(s_o, axis=1)
    s_o = K.expand_dims(s_o, axis=1)
    for h in range(height-1):
        s_map = K.concatenate([s_map, s_o], axis=1)

    ''' frame-level temporal attention '''
    t_o = K.l2_normalize(t_o, axis=1)
    t_map = K.expand_dims(t_o, axis=2)
    t_o = K.expand_dims(t_o, axis=2)
    for w in range(width-1):
        t_map = K.concatenate([t_map, t_o], axis=2)

    ''' Prior Spatial Attention '''
    a_o = multiply([s_map, t_map])
    roi_map_value = np.zeros((1, 300, 25))
    for i in range(300):
        for j in [3,8,12,14,17,19]:
            roi_map_value[0, i, j] = 1
    roi_map = K.variable(roi_map_value)
    a_o = Add()([a_o, roi_map])

    a_map = K.expand_dims(a_o, axis=3)
    a_o = K.expand_dims(a_o, axis=3)
    for c in range(channels-1):
        a_map = K.concatenate([a_map, a_o], axis=3)

    out = multiply([a_map, ipt])
    return out

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=2, name=bn_name)(x)
    return x

def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def network_with_attention(height, width, channels, classes):

    inpt = Input(shape=(height, width, channels))

    # adaptive spatial attention
    x = Conv2D(64, (15,1), padding='valid', strides=(1,1), activation='relu')(inpt)
    x = BatchNormalization(axis=2)(x)
    x = MaxPooling2D(pool_size=(15,1), strides=(1,1), padding='same')(x)
    x = Flatten()(x)
    spatial_output = Dense(width)(x)

    # frame-level temperal attention
    y = Reshape((height, -1))(inpt)
    y = LSTM(height)(y)
    temperal_output = Dense(height)(y)

    # ST-MAP layer
    #z = Lambda(attention_map)([spatial_output, temperal_output, inpt, channels])
    z = attention_map([spatial_output, temperal_output, inpt, channels])

    # x_plu: block-level temporal attention
    inpt_x = Input(shape=(height, 1))
    z = X_plus_Layer()([inpt_x, z])

    # ---------------- FC ------------------
    # z = Flatten()(z)
    # z = Dense(128)(z)
    # z = Dense(32)(z)
    # class_result = Dense(1, activation='sigmoid')(z)

    # ---------------- LeNet 5 -----------------
    # z = ZeroPadding2D((1,1))(z)
    # z = Conv2D(6,(5,5),strides=(1,1),padding='valid', activation='relu')(z)
    # z = MaxPooling2D((2,2),strides=(2,2))(z)
    # z = Conv2D(6,(5,5),strides=(1,1),padding='valid', activation='relu')(z)
    # z = MaxPooling2D((2,2),strides=(2,2))(z)
    # z = Flatten()(z)
    # z = Dense(120,activation='relu')(z)
    # z = Dense(84,activation='relu')(z)
    # class_result = Dense(1,activation='sigmoid')(z)

    # ---------------- ResNet 18 -----------------
    res = ZeroPadding2D((3, 3))(z)

    # conv1
    res = Conv2d_BN(res, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    res = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(res)

    # conv2_x
    '''res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))

    # conv3_x
    res = identity_Block(res, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=128, kernel_size=(3, 3))

    # conv4_x
    res = identity_Block(res, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=256, kernel_size=(3, 3))

    # conv5_x
    res = identity_Block(res, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=512, kernel_size=(3, 3))

    res = AveragePooling2D(pool_size=(7, 1))(res)'''
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"
    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
        x = Activation('relu', name=s_id + relu + sq1x1)(x)

        left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
        left = Activation('relu', name=s_id + relu + exp1x1)(left)

        right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
        right = Activation('relu', name=s_id + relu + exp3x3)(right)

        x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
        return x

    #x = fire_module(res, fire_id=2, squeeze=16, expand=64)
    #x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    #x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(x)
    # conv2_x
    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))

    x = fire_module(res, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool5')(x)

    # conv3_x
    res = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=128, kernel_size=(3, 3))

    x = fire_module(res, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)

    # conv4_x
    res = identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=256, kernel_size=(3, 3))

    x = fire_module(res, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    # conv5_x
    res = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=512, kernel_size=(3, 3))

    res = AveragePooling2D(pool_size=(7, 1), padding='same')(res)

    res = Flatten()(res)
    res=Dense(576)(res)
    print('res',res.shape)
    a=att(shape=(height, width, channels))
    b=a(z)
    print(b.shape,'a')
   # a=tf.squeeze(a,axis=1)
    res=add([res, b])
    res = Dense(1, name='resnet_result')(res)
    print(res.shape)
    class_result = Activation('sigmoid')(res)
    print(class_result)
    model = Model(inputs=[inpt, inpt_x], outputs=class_result)
    return model

def to_be_2d(y):
    ny = np.zeros((len(y), 2))
    for i in range(len(y)):
        if y[i] == 0:
            ny[i,0] = 1
        elif y[i] == 1:
            ny[i,1] = 1
    return ny

data_path = "D:/Datasets/CelebDF v2/6_Train_Data_subset/"
save_path = "D:/Datasets/CelebDF v2/7_model_v2/"

best_acc = 0
if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    #batch_size = 32
    batch_size = 32
    #epochs = 10000
    epochs = 100

    try:
        # load data
        data_dir = data_path
        mit_list = np.load(data_dir+"mit.npy")
        Meso_list = np.load(data_dir+"Meso.npy")
        y_list = np.load(data_dir+"y.npy")
        Meso_list = np.reshape(Meso_list, (len(y_list), 300, 1))
    except IOError:
        print("No data")
        sys.exit(0)

    model = network_with_attention(300, 25, 3, 2)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
    opt=keras.optimizers.Adam(lr_schedule)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.summary()

    test_num = 50
    test_st = int((len(y_list)-test_num)/2)
    test_ed = test_st + test_num
    test = [i for i in range(test_st, test_ed)]
    train = []
    for i in range(len(y_list)):
        if i in test:
            continue
        train.append(i)

    x_train_mit = mit_list[train]
    x_train_meso = Meso_list[train]
    y_train = y_list[train]

    x_test_mit = mit_list[test]
    x_test_meso = Meso_list[test]
    y_test = y_list[test]

    best_acc = 0
    def saveModel(epoch, logs):
        score, acc = model.evaluate([x_test_mit, x_test_meso], y_test,
                            batch_size=batch_size)
        global best_acc
        # val_acc = logs['val_accuracy']
        # t_acc = logs['accuracy']
        if acc > best_acc:
            print("Save model, acc=", acc)
            best_acc = acc
            model.save_weights(save_path + 'PK_Meso_FM.h5')
    callbacks = [LambdaCallback(on_epoch_end=saveModel)]

    history=model.fit([x_train_mit, x_train_meso], y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=callbacks) # callbacks=[PredictionCallback()]
    #model.save_weights(save_path + 'PK_Meso.h5')
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    num_epochs=epochs
    epochs = range(1,num_epochs+1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    #plt.savefig('D:/Sem VI - Textbooks & Materials/CAPSTONE PROJECT/DeepRhythm_Code/Capstone Arch Codes/fdftnet_fm/train_val_loss.png', dpi=600)
    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    epochs = range(1,num_epochs+1)
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'r', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    #plt.savefig('D:/Sem VI - Textbooks & Materials/CAPSTONE PROJECT/DeepRhythm_Code/Capstone Arch Codes/fdftnet_fm/train_val_acc.png', dpi=600)
