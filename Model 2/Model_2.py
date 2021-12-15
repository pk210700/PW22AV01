import keras
import tensorflow as tf
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
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras.utils.vis_utils import plot_model

from keras import backend as K



def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


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
    
    #x = fire_module(res, fire_id=2, squeeze=16, expand=64)
    #x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    #x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(x)
    # conv2_x
    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))
    
  #  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool5')(x)

    # conv3_x
    res = identity_Block(res, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=128, kernel_size=(3, 3))
    
    res=_inverted_residual_block(res,16,(3,3),1,1.0,1,1)
    res=_inverted_residual_block(res,24,(3,3),6,1.0,2,2)

    # conv4_x
    res = identity_Block(res, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=256, kernel_size=(3, 3))
    
    res=_inverted_residual_block(res,32,(3,3),6,1.0,2,3)
    res=_inverted_residual_block(res,64,(3,3),6,1.0,2,4)

    # conv5_x
    res = identity_Block(res, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=512, kernel_size=(3, 3))
    
    res=_inverted_residual_block(res,96,(3,3),6,1.0,1,3)
    res=_inverted_residual_block(res,160,(3,3),6,1.0,2,3)
    
    res=_inverted_residual_block(res,320,(3,3),6,1.0,1,1)
   # res=_inverted_residual_block(res,,(3,3),6,1.0,2,4)
   
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

data_path = "C:/Users/bhatn/OneDrive/Documents/6_Train_Data/"
save_path = "C:/Users/bhatn/OneDrive/Documents/7_Model/FDFTNETTrialFM/"

best_acc = 0
if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    #batch_size = 32
    batch_size = 10
    #epochs = 10000
    epochs = 10

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

    test_num = 0
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
            model.save(save_path + 'PK_Meso.h5')
    callbacks = [LambdaCallback(on_epoch_end=saveModel)]

    model.fit([x_train_mit, x_train_meso], y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)#, callbacks=callbacks) # callbacks=[PredictionCallback()]
    model.save_weights(save_path + 'PK_Meso.h5')



