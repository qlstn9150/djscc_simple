import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Conv2D, Cropping2D, concatenate, Dense
from tensorflow.keras.layers import Input, Layer, UpSampling2D, Flatten, Conv2DTranspose
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model



def normalize_pixels(train_data, test_data):
	train_norm = train_data.astype('float32')
	test_norm = test_data.astype('float32')
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm





def basic750(comp_ratio, F=5):
    input_images = Input(shape=(32, 32, 3))
    ############################### Buliding Encoder ##############################
    # 1st convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal')(
        input_images)
    prelu1 = PReLU()(conv1)
    # 2nd convolutional layer
    conv2 = Conv2D(filters=80, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal')(prelu1)
    prelu2 = PReLU()(conv2)
    # 3rd convolutional layer
    conv3 = Conv2D(filters=50, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(prelu2)
    prelu3 = PReLU()(conv3)
    # 4th convolutional layer
    conv4 = Conv2D(filters=40, kernel_size=(F,F), strides=1, padding='same', kernel_initializer='he_normal')(prelu3)
    prelu4 = PReLU()(conv4)
    # 5th convolutional layer
    c = Calculate_filters(comp_ratio, F)
    conv5 = Conv2D(filters=c, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(prelu4)
    encoder = PReLU()(conv5)

    real_prod = NormalizationNoise()(encoder)

    ############################### Building Decoder ##############################

    decoder = Conv2DTranspose(filters=40, kernel_size=(5,5), strides=1, padding='same',
                              kernel_initializer='he_normal')(real_prod)
    decoder = PReLU()(decoder)
    decoder = Conv2DTranspose(filters=50, kernel_size=(5,5), strides=1, padding='same',
                              kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)

    decoder = Conv2DTranspose(filters=80, kernel_size=(5,5), strides=1, padding='same',
                              kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    decoder = Conv2DTranspose(filters=16, kernel_size=(5,5), strides=2, padding='valid',
                              kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    # decoder_up = UpSampling2D((2,2))(decoder)
    decoder = Conv2DTranspose(filters=3, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal',
                              activation='sigmoid')(decoder)
    # decoder = PReLU()(decoder)
    decoder_up = UpSampling2D((2, 2))(decoder)
    decoder = Cropping2D(cropping=((13, 13), (13, 13)))(decoder_up)

    ############################### Buliding Models ###############################
    model = Model(input_images, decoder)
    return model

def basic(snr, comp_ratio, F=5):
    input_images = Input(shape=(32, 32, 3))
    ############################### Buliding Encoder ##############################
    # 1st convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal')(
        input_images)
    prelu1 = PReLU()(conv1)
    # 2nd convolutional layer
    conv2 = Conv2D(filters=80, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal')(prelu1)
    prelu2 = PReLU()(conv2)
    # 3rd convolutional layer
    conv3 = Conv2D(filters=50, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(prelu2)
    prelu3 = PReLU()(conv3)
    # 4th convolutional layer
    conv4 = Conv2D(filters=40, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(prelu3)
    prelu4 = PReLU()(conv4)
    # 5th convolutional layer
    c = Calculate_filters(comp_ratio, F)
    conv5 = Conv2D(filters=c, kernel_size=(5,5), strides=1, padding='same', kernel_initializer='he_normal')(prelu4)
    encoder = PReLU()(conv5)

    real_prod = NormalizationNoise(snr)(encoder)

    ############################### Building Decoder ##############################

    decoder = Conv2DTranspose(filters=40, kernel_size=(5,5), strides=1, padding='same',
                              kernel_initializer='he_normal')(real_prod)
    decoder = PReLU()(decoder)
    decoder = Conv2DTranspose(filters=50, kernel_size=(5,5), strides=1, padding='same',
                              kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)

    decoder = Conv2DTranspose(filters=80, kernel_size=(5,5), strides=1, padding='same',
                              kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    decoder = Conv2DTranspose(filters=16, kernel_size=(5,5), strides=2, padding='valid',
                              kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)
    # decoder_up = UpSampling2D((2,2))(decoder)
    decoder = Conv2DTranspose(filters=3, kernel_size=(5,5), strides=2, padding='valid', kernel_initializer='he_normal',
                              activation='sigmoid')(decoder)
    # decoder = PReLU()(decoder)
    decoder_up = UpSampling2D((2, 2))(decoder)
    decoder = Cropping2D(cropping=((13, 13), (13, 13)))(decoder_up)

    ############################### Buliding Models ###############################
    model = Model(input_images, decoder)
    return model

def model6(comp_ratio, F=5):
    input_images = Input(shape=(32, 32, 3))
    ############################### Building Decoder ##############################
    conv1 = Conv2D(filters=16, kernel_size=(5, 5), strides=1,
                   padding='same', kernel_initializer='he_normal')(input_images)
    prelu1 = PReLU()(conv1)

    conv2 = Conv2D(filters=80, kernel_size=(5, 5), strides=1,
                   padding='same', kernel_initializer='he_normal')(prelu1)
    prelu2 = PReLU()(conv2)

    conv3 = Conv2D(filters=50, kernel_size=(5, 5), strides=1,
                   padding='same', kernel_initializer='he_normal')(prelu2)
    prelu3 = PReLU()(conv3)

    conv4 = Conv2D(filters=40, kernel_size=(F, F), strides=1,
                   padding='same', kernel_initializer='he_normal')(prelu3)
    prelu4 = PReLU()(conv4)

    conv5 = Conv2D(filters=Calculate_filters(comp_ratio, F), kernel_size=(5, 5), strides=1,
                   padding='same', kernel_initializer='he_normal')(prelu4)
    encoder = PReLU()(conv5)

    ############################### NOISE ##############################
    real_prod = NormalizationNoise()(encoder)

    ############################### Building Decoder ##############################
    d1 = Conv2D(filters=40, kernel_size=(5, 5), strides=1,
                padding='same', kernel_initializer='he_normal')(real_prod)
    d1 = PReLU()(d1)

    d2 = Conv2D(filters=50, kernel_size=(5, 5), strides=1,
                padding='same', kernel_initializer='he_normal')(d1)
    d2 = PReLU()(d2)

    d3 = Conv2D(filters=80, kernel_size=(5, 5), strides=1,
                     padding='same', kernel_initializer='he_normal')(d2)
    d3 = PReLU()(d3)

    d4 = Conv2D(filters=16, kernel_size=(5, 5), strides=1,
                     padding='same', kernel_initializer='he_normal')(d3)
    d4 = PReLU()(d4)

    d_output = Conv2D(filters=3, kernel_size=(5, 5), strides=1,
                     padding='same', kernel_initializer='he_normal',
                     activation='sigmoid')(d4)

    ############################### Buliding Models ###############################
    model = Model(input_images, d_output)
    return model

# change encoder + symmetric
def model1(snr, comp_ratio, F=5):
    input_images = Input(shape=(32, 32, 3))

    e1= Conv2D(filters=40, kernel_size=(5,5), strides=1,
               padding='same', kernel_initializer='he_normal')(input_images)
    e1 = PReLU()(e1)

    c = Calculate_filters(comp_ratio, F)
    e2 = Conv2D(filters=c, kernel_size=(5,5), strides=1,
                padding='same', kernel_initializer='he_normal')(e1)
    e_output = PReLU(name='e_output')(e2)

    real_prod = NormalizationNoise(snr)(e_output)

    ############################### Building Decoder ##############################
    d1 = Conv2D(filters=40, kernel_size=(5,5), strides=1,
                padding='same', kernel_initializer='he_normal')(real_prod)
    d1 = PReLU()(d1)

    d2= Conv2D(filters=3, kernel_size=(5,5), strides=1,
               padding='same', kernel_initializer='he_normal')(d1)
    d_output = PReLU()(d2)

    ############################### Buliding Models ###############################
    model = Model(input_images, d_output)
    return model

# change decoder(simple)
def model8(snr, comp_ratio, F=5):
    input_images = Input(shape=(32, 32, 3), name='input')
    c = Calculate_filters(comp_ratio, F)

    e1= Conv2D(filters=40, kernel_size=(5,5), strides=1,
               padding='same', kernel_initializer='he_normal')(input_images)
    e1 = PReLU()(e1)

    e2 = Conv2D(filters=c, kernel_size=(5,5), strides=1,
                padding='same', kernel_initializer='he_normal')(e1)
    e_output = PReLU(name='e_output')(e2)

    ############################### NOISE ##############################
    c_output = NormalizationNoise(snr)(e_output)
    ############################### Building Decoder ##############################

    d1 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(c_output)
    d1 = PReLU()(d1)

    d2 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(d1)
    d2 = PReLU()(d2)

    d3 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal')(d2)
    d3 = PReLU()(d3)

    d4 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal')(d3)
    d4 = PReLU()(d4)

    d_output = Conv2D(filters=3, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal',
                     activation='sigmoid', name='d_output')(d4)

    ############################### Buliding Models ###############################
    model = Model(input_images, d_output)
    return model

#ResNet
def model9(snr, comp_ratio, F=5):
    input_images = Input(shape=(32, 32, 3), name='input')

    e1= Conv2D(filters=40, kernel_size=(5,5), strides=1,
               padding='same', kernel_initializer='he_normal')(input_images)
    e1 = PReLU()(e1)

    c = Calculate_filters(comp_ratio, F)
    e2 = Conv2D(filters=c, kernel_size=(5,5), strides=1,
                padding='same', kernel_initializer='he_normal')(e1)
    e_output = PReLU(name='e_output')(e2)

    ############################### NOISE ##############################
    c_output = NormalizationNoise(snr)(e_output)
    ############################### Building Decoder ##############################

    d1 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(c_output)
    d1 = PReLU()(d1)

    d2 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(d1)
    d2 = PReLU()(d2)
    d2 = Add()([d1, d2])

    d3 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal')(d2)
    d3 = PReLU()(d3)
    d3 = Add()([d1, d2, d3])

    d4 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal')(d3)
    d4 = PReLU()(d4)
    d4 = Add()([d1, d2, d3, d4])

    d_output = Conv2D(filters=3, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal',
                     activation='sigmoid', name='d_output')(d4)

    ############################### Buliding Models ###############################
    model = Model(input_images, d_output)
    return model

#DesneNet
def model10(snr, comp_ratio, F=5):
    input_images = Input(shape=(32, 32, 3), name='input')

    e1= Conv2D(filters=40, kernel_size=(5,5), strides=1,
               padding='same', kernel_initializer='he_normal')(input_images)
    e1 = PReLU()(e1)

    c = Calculate_filters(comp_ratio, F)
    e2 = Conv2D(filters=c, kernel_size=(5,5), strides=1,
                padding='same', kernel_initializer='he_normal')(e1)
    e_output = PReLU(name='e_output')(e2)

    ############################### NOISE ##############################
    c_output = NormalizationNoise(snr)(e_output)
    ############################### Building Decoder ##############################

    d1 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(c_output)
    d1 = PReLU()(d1)

    d2 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(d1)
    d2 = PReLU()(d2)
    d2 = concatenate([d1, d2])

    d3 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal')(d2)
    d3 = PReLU()(d3)
    d3 = concatenate([d1, d2, d3])

    d4 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal')(d3)
    d4 = PReLU()(d4)
    d4 = concatenate([d1, d2, d3, d4])

    d_output = Conv2D(filters=3, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal',
                     activation='sigmoid', name='d_output')(d4)

    ############################### Buliding Models ###############################
    model = Model(input_images, d_output)
    return model

#change decoder
def model4(snr, comp_ratio, F=5):
    ############################### Buliding Encoder ##############################
    input_images = Input(shape=(32, 32, 3))

    e1= Conv2D(filters=40, kernel_size=(5,5), strides=1,
               padding='same', kernel_initializer='he_normal')(input_images)
    e1 = PReLU()(e1)

    c = Calculate_filters(comp_ratio, F)
    e2 = Conv2D(filters=c, kernel_size=(5,5), strides=1,
                padding='same', kernel_initializer='he_normal')(e1)
    e_output = PReLU(name='e_output')(e2)
    ################################ NOISE CHANNEL ################################
    real_prod = NormalizationNoise(snr)(e_output)
    ############################### Building Decoder ##############################
    d1 = Conv2D(16, (5, 5), strides=1,
                padding='same', kernel_initializer='he_normal')(real_prod)
    d1 = BatchNormalization()(d1)
    d1= PReLU()(d1)

    d2 = Conv2D(40, (5, 5), strides=1,
                padding='same', kernel_initializer='he_normal')(d1)
    d2 = BatchNormalization()(d2)
    d2= PReLU()(d2)

    d3 = Conv2D(40, (5, 5), strides=1,
                padding='same', kernel_initializer='he_normal')(d2)
    d3 = BatchNormalization()(d3)
    d3= PReLU()(d3)

    d3 = Add()([d2, d3])

    d_output = Conv2D(3, (5, 5), strides=1,
                      padding='same', kernel_initializer='he_normal',
                      activation='sigmoid')(d3)
    ############################### Buliding Models ###############################
    model = Model(input_images, d_output)
    return model

#filter 1X1
def model11(snr, comp_ratio, F=1):
    input_images = Input(shape=(32, 32, 3), name='input')
    c = Calculate_filters(comp_ratio, F)

    e1= Conv2D(filters=50, kernel_size=(1,1), strides=1,
               padding='same', kernel_initializer='he_normal')(input_images)
    e1 = PReLU()(e1)

    e2 = Conv2D(filters=c, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(e1)
    e_output = PReLU(name='e_output')(e2)

    ############################### NOISE ##############################
    c_output = NormalizationNoise(snr)(e_output)
    ############################### Building Decoder ##############################

    d1 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(c_output)
    d1 = PReLU()(d1)

    d2 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(d1)
    d2 = PReLU()(d2)

    d3 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal')(d2)
    d3 = PReLU()(d3)

    d4 = Conv2D(filters=50, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal')(d3)
    d4 = PReLU()(d4)

    d_output = Conv2D(filters=3, kernel_size=(1,1), strides=1,
                     padding='same', kernel_initializer='he_normal',
                     activation='sigmoid', name='d_output')(d4)

    ############################### Buliding Models ###############################
    model = Model(input_images, d_output)
    return model

def model12(snr, comp_ratio, F=1):
    input_images = Input(shape=(32, 32, 3), name='input')
    c = Calculate_filters(comp_ratio, F)

    e1= Conv2D(filters=50, kernel_size=(1,1), strides=1,
               padding='same', kernel_initializer='he_normal')(input_images)
    e1 = PReLU()(e1)

    e2 = Conv2D(filters=c, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(e1)
    e_output = PReLU(name='e_output')(e2)

    ############################### NOISE ##############################
    c_output = NormalizationNoise()(e_output)
    ############################### Building Decoder ##############################
    decoder = Conv2DTranspose(filters=50, kernel_size=(1,1), strides=1,
                              padding='same', kernel_initializer='he_normal')(c_output)
    decoder = PReLU()(decoder)

    # decoder_up = UpSampling2D((2,2))(decoder)
    decoder = Conv2DTranspose(filters=3, kernel_size=(1,1), strides=1,
                              padding='same', kernel_initializer='he_normal',
                              activation='sigmoid')(decoder)
    # decoder = PReLU()(decoder)
    #decoder_up = UpSampling2D((2, 2))(decoder)
    #decoder = Cropping2D(cropping=((13, 13), (13, 13)))(decoder_up)

    ############################### Buliding Models ###############################
    model = Model(input_images, decoder)
    return model

def model13(snr, comp_ratio, F=1):
    input_images = Input(shape=(32, 32, 3), name='input')
    c = Calculate_filters(comp_ratio, F)

    e1= Conv2D(filters=32, kernel_size=(1,1), strides=1,
               padding='same', kernel_initializer='he_normal')(input_images)
    e1 = PReLU()(e1)

    e2 = Conv2D(filters=c, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(e1)
    e_output = PReLU(name='e_output')(e2)

    ############################### NOISE ##############################
    c_output = NormalizationNoise(snr)(e_output)
    ############################### Building Decoder ##############################
    decoder = Conv2DTranspose(filters=32, kernel_size=(1,1), strides=1,
                              padding='same', kernel_initializer='he_normal')(c_output)
    decoder = PReLU()(decoder)

    # decoder_up = UpSampling2D((2,2))(decoder)
    decoder = Conv2DTranspose(filters=3, kernel_size=(1,1), strides=1,
                              padding='same', kernel_initializer='he_normal',
                              activation='sigmoid')(decoder)
    # decoder = PReLU()(decoder)
    #decoder_up = UpSampling2D((2, 2))(decoder)
    #decoder = Cropping2D(cropping=((13, 13), (13, 13)))(decoder_up)

    ############################### Buliding Models ###############################
    model = Model(input_images, decoder)
    return model

def model14(snr, comp_ratio, F=1):
    input_images = Input(shape=(32, 32, 3), name='input')
    c = Calculate_filters(comp_ratio, F)

    e1= Conv2D(filters=32, kernel_size=(1,1), strides=1,
               padding='same', kernel_initializer='he_normal')(input_images)
    e1 = PReLU()(e1)

    e2= Conv2D(filters=32, kernel_size=(1,1), strides=1,
               padding='same', kernel_initializer='he_normal')(e1)
    e2 = PReLU()(e2)

    e3 = Conv2D(filters=c, kernel_size=(1,1), strides=1,
                padding='same', kernel_initializer='he_normal')(e2)
    e_output = PReLU(name='e_output')(e3)

    ############################### NOISE ##############################
    c_output = NormalizationNoise(snr)(e_output)
    ############################### Building Decoder ##############################
    decoder = Conv2DTranspose(filters=32, kernel_size=(1,1), strides=1,
                              padding='same', kernel_initializer='he_normal')(c_output)
    decoder = PReLU()(decoder)

    decoder = Conv2DTranspose(filters=32, kernel_size=(1,1), strides=1,
                              padding='same', kernel_initializer='he_normal')(decoder)
    decoder = PReLU()(decoder)

    decoder = Conv2DTranspose(filters=3, kernel_size=(1,1), strides=1,
                              padding='same', kernel_initializer='he_normal',
                              activation='sigmoid')(decoder)


    ############################### Buliding Models ###############################
    model = Model(input_images, decoder)
    return model