import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
tf.config.list_physical_devices('GPU')

from tensorflow.keras.datasets import cifar10
from train_eval import *
from plot import *

(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)

model = 'basic750'
model_f = basic750
snr_train = [0, 10, 20]
snr_test = [2, 4, 7, 10, 13, 16, 18, 22, 25, 27] #[2, 10, 18, 26]
compression_ratios = [0.06, 0.26, 0.49] #0.06, 0.26, 0.49

### TRAIN & EVALAUATION ###
train(model, model_f, snr_train, compression_ratios, x_train, x_test, batch_size=100, epochs=20)
comp_eval(model, x_test, testX, compression_ratios, snr_train)
test_eval(model, x_test, testX, compression_ratios, snr_train, snr_test)

### VISUALIZATION ###
#all_img(model)
model_list = ['basic']
#comp_plot('psnr', model_list, snr_train)
#comp_plot('psnr', model_list, snr_train)
#test_plot(model_list, snr_train, compression_ratios)

### compare many models ###
#model_list = ['basic', model]
#all_model_compare('psnr', model_list, snr_train)
#all_model_compare('ssim', model_list, snr_train)




