import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from model import *


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

            psnr = peak_signal_noise_ratio(testX, pred_images)
            ssim = structural_similarity(testX, pred_images, multichannel=True)

            model_dic['PSNR'].append(psnr)
            model_dic['SSIM'].append(ssim)
            print('Comp_Ratio = ', comp_ratio)
            print('PSNR = ', psnr)
            print('SSIM = ', ssim)
            print('\n')

        os.makedirs('./result_txt/plot1/{0}'.format(model), exist_ok=True)
        path = './result_txt/plot1/{0}/SNR{1}.txt'.format(model, snr)
        with open(path, 'w') as f:
            print(compression_ratios, '\n', model_dic['PSNR'], '\n', model_dic['SSIM'], file=f)
        f.closed

    original_images = array_to_img(x_test[6,])
    original_images.save('./img/original.jpg')

def test_eval(model, x_test, testX, compression_ratios, snr_train, snr_test):
    for comp_ratio in compression_ratios:
        for snr in snr_train:
            model_dic = {'Test_snr': [], 'PSNR': [], 'SSIM':[]}
            for snr_t in snr_test:
                tf.keras.backend.clear_session()
                path = './checkpoints/{0}/CompRatio{1}_SNR{2}.h5'.format(model, comp_ratio, snr)
                autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
                K.set_value(autoencoder.get_layer('normalization_noise').snr_db, snr_t)

                pred_images = autoencoder.predict(x_test) * 255
                pred_images = pred_images.astype('uint8')

                psnr = peak_signal_noise_ratio(testX, pred_images)
                ssim = structural_similarity(testX, pred_images, multichannel=True)
                model_dic['Test_snr'].append(snr_t)
                model_dic['PSNR'].append(psnr)
                model_dic['SSIM'].append(ssim)
                print('Test SNR =  ', snr_t)
                print('PSNR = ', psnr)
                print('SSIM = ', ssim)
                print('\n')

            os.makedirs('./result_txt/plot2/{0}'.format(model), exist_ok=True)
            path = './result_txt/plot2/{0}/CompRatio{1}_SNR{2}.txt'.format(model, comp_ratio, snr)
            with open(path, 'w') as f:
                print(snr_test, '\n', model_dic['PSNR'], '\n', model_dic['SSIM'], file=f)
            f.closed