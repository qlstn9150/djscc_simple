import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.config.list_physical_devices('GPU')

import os
import time
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.optimizers import adam_v2
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from skimage.metrics import peak_signal_noise_ratio

from model import ModelCheckponitsHandler

#normalizing the training and test data
def normalize_pixels(train_data, test_data):
    #convert integer values to float
	train_norm = train_data.astype('float32')
	test_norm = test_data.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm


def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)



def train(model_str, model_f, snr_train, compression_ratios, x_train, x_test, batch_size, epochs):
    for snr in snr_train:
        for comp_ratio in compression_ratios:
            tf.keras.backend.clear_session()

            # load model
            model, c = model_f(comp_ratio)
            K.set_value(model.get_layer('normalization_noise').snr_db, snr)

            model.compile(optimizer=adam_v2.Adam(learning_rate=0.0001), loss='mse', metrics=[psnr]) #accuracy

            model.summary()
            print('\t-----------------------------------------------------------------')
            print('\t|\t\t\t\t\t\t\t\t|')
            print('\t|\t\t\t\t\t\t\t\t|')
            print('\t| Training Parameters: Filter Size: {0}, Compression ratio: {1} |'.format(c, comp_ratio))
            print('\t|\t\t\t  SNR: {0} dB\t\t\t\t|'.format(snr))
            print('\t|\t\t\t\t\t\t\t\t|')
            print('\t|\t\t\t\t\t\t\t\t|')
            print('\t-----------------------------------------------------------------')

            # callbacks
            os.makedirs('./Tensorboard/{0}'.format(model_str), exist_ok=True)
            os.makedirs('./checkpoints/{0}'.format(model_str), exist_ok=True)

            tb = TensorBoard(log_dir='./Tensorboard/{0}/CompRatio{1}_SNR{2}'.format(model_str, str(comp_ratio), str(snr)))

            checkpoint = ModelCheckpoint(filepath='./checkpoints/{0}/CompRatio{1}_SNR{2}.h5'.format(model_str, str(comp_ratio), str(snr)),
                                         monitor = 'val_loss', save_best_only = True)

            ckpt = ModelCheckponitsHandler(model_str, comp_ratio, snr, model, step=50)
            earlystop = EarlyStopping(monitor='loss', patience=10)


            # train model
            start = time.perf_counter()
            model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=epochs,
                      callbacks=[tb, checkpoint, ckpt, earlystop], validation_data=(x_test, x_test))
            end = time.perf_counter()



            print('The NN has trained ' + str(end - start) + ' s')
            print('============ FINISH {0}_CompRation{1}_SNR{2} ============'.format(model_str, comp_ratio, snr))
            print('\n')
            print('\n')
