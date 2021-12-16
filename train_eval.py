import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from model import *

def train(model_str, model_f, all_snr, compression_ratios, x_train, x_test, batch_size, epochs):
    for snr in all_snr:
        for comp_ratio in compression_ratios:
            tf.keras.backend.clear_session()

            # load model
            model = model_f(snr, comp_ratio)
            model.summary()

            os.makedirs('./Tensorboard/{0}'.format(model_str), exist_ok=True)
            os.makedirs('./checkpoints/{0}'.format(model_str), exist_ok=True)

            tb = TensorBoard(log_dir='./Tensorboard/{0}/CompRatio{1}_SNR{2}'.format(model_str, str(comp_ratio), str(snr)))

            checkpoint = ModelCheckpoint(filepath='./checkpoints/{0}/CompRatio{1}_SNR{2}.h5'.format(model_str, str(comp_ratio), str(snr)),
                                         monitor = 'val_loss', save_best_only = True)

            ckpt = ModelCheckponitsHandler(model_str, comp_ratio, snr, model, step=50)
            #earlystop = EarlyStopping(monitor='accuracy', patience=5)

            K.set_value(model.get_layer('normalization_noise').snr_db, snr)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])


            # train model
            start = time.perf_counter()
            model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=epochs,
                      callbacks=[tb, checkpoint, ckpt], validation_data=(x_test, x_test))
            end = time.perf_counter()

            print('The NN has trained ' + str(end - start) + ' s')
            print('============ FINISH {0}_CompRation{1}_SNR{2} ============'.format(model_str, comp_ratio, snr))
            print('\n')
            print('\n')

#save image
def comp_eval(model, x_test, testX, compression_ratios, snr_train):
    for snr in snr_train:
        model_dic = {'Pred_Images': [], 'PSNR': [], 'SSIM': []}
        for comp_ratio in compression_ratios:
            tf.keras.backend.clear_session()
            print('==============={0}_CompRation{1}_SNR{2}============'.format(model, comp_ratio, snr))
            path = './checkpoints/{0}/CompRatio{1}_SNR{2}.h5'.format(model, comp_ratio, snr)
            autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
            K.set_value(autoencoder.get_layer('normalization_noise').snr_db, snr)

            pred_images = autoencoder.predict(x_test) * 255
            pred_images = pred_images.astype('uint8')

            os.makedirs('./img/{0}'.format(model), exist_ok=True)
            sample_images = array_to_img(pred_images[6,])
            sample_images.save('./img/{0}/pred_CompRatio{1}_SNR{2}.jpg'.format(model, comp_ratio, snr))

            ssim = structural_similarity(testX, pred_images, multichannel=True)
            psnr = peak_signal_noise_ratio(testX, pred_images)

            model_dic['PSNR'].append(psnr)
            model_dic['SSIM'].append(ssim)
            print('Comp_Ratio = ', comp_ratio)
            print('PSNR = ', psnr)
            print('SSIM = ', ssim)
            print('\n')

        os.makedirs('./result_txt/plot1', exist_ok=True)
        path = './result_txt/plot1/{0}_SNR{1}.txt'.format(model, snr)
        with open(path, 'w') as f:
            print(compression_ratios, '\n', model_dic['PSNR'], '\n', model_dic['SSIM'], file=f)
        f.closed

    original_images = array_to_img(x_test[6,])
    original_images.save('./img/original.jpg')


def test_eval(model, x_test, testX, compression_ratios, snr_train, snr_test):
    for comp_ratio in compression_ratios:
        for snr in snr_train:
            model_dic = {'Test_snr': [], 'PSNR': []}
            for snr_t in snr_test:
                tf.keras.backend.clear_session()
                path = './checkpoints/{0}/CompRatio{1}_SNR{2}.h5'.format(model, comp_ratio, snr)
                autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
                K.set_value(autoencoder.get_layer('normalization_noise').snr_db, snr_t)

                pred_images = autoencoder.predict(x_test) * 255
                pred_images = pred_images.astype('uint8')
                psnr = peak_signal_noise_ratio(testX, pred_images)
                model_dic['Test_snr'].append(snr_t)
                model_dic['PSNR'].append(psnr)
                print('Test SNR = ', snr_t)
                print('PSNR = ', psnr)
                print('\n')

            os.makedirs('./result_txt/plot2', exist_ok=True)
            path = './result_txt/plot2/{0}_CompRatio{1}_SNR{2}.txt'.format(model, comp_ratio, snr)
            with open(path, 'w') as f:
                print(snr_test, '\n', model_dic['PSNR'], file=f)
            f.closed

