import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
tf.config.list_physical_devices('GPU')

from keras.datasets import cifar10

from model3 import *
from train import *
from evaluation import *
from plot import *

################### DATA ########################
(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)


#################### RUN ########################
model = 'model10'
model_f = model10
snr_train = [0, 10, 20]
snr_test =  [2, 4, 7, 10, 13, 16, 18, 22, 25, 27] #[2, 10, 18, 26]
compression_ratios = [0.06, 0.26, 0.49]

### TRAIN & EVALAUATION ###
train(model, model_f, snr_train, compression_ratios, x_train, x_test, batch_size=100, epochs=20)
comp_eval(model, x_test, testX, compression_ratios, snr_train)
test_eval(model, x_test, testX, compression_ratios, snr_train, snr_test)

### VISUALIZATION ###
model_list = ['DJSCC',  model]
comp_plot('psnr', model_list, snr_train)
test_plot('psnr',model_list, snr_train, compression_ratios=[0.49])
comp_plot('ssim', model_list, snr_train)
test_plot('ssim',model_list, snr_train, compression_ratios=[0.49])
#all_img(model)

### compare many models ###
model_list = ['DJSCC', 'DJSCC-Advanced', model]
all_model_compare('psnr', model_list, snr_train)
all_model_compare('ssim', model_list, snr_train)