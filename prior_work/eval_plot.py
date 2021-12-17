import json
import glob
import cv2

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from AutoencoderModel import *


### DATA ###
def normalize_pixels(train_data, test_data):
    #convert integer values to float
	train_norm = train_data.astype('float32')
	test_norm = test_data.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)

### EVALUATION ###
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


### PLOT ###
def comp_plot(metric, model_str, snr_train):
    colors = list(mcolors.TABLEAU_COLORS)
    markers = ['o', 's']
    ls = ['--', '-']
    i = 0
    for model in model_str:
        j = 0
        for snr in snr_train:
            path = './result_txt/plot1/{0}_SNR{1}.txt'.format(model, snr)
            with open(path, 'r') as f:
                text = f.read()
                compression_ratios = text.split('\n')[0]
                compression_ratios = json.loads(compression_ratios)
                if metric == 'psnr':
                    psnr = text.split('\n')[1]
                    metric_score = json.loads(psnr)
                else:
                    ssim = text.split('\n')[2]
                    metric_score = json.loads(ssim)
            label = '{0} (SNR={1}dB)'.format(model, snr)
            plt.plot(compression_ratios, metric_score, ls=ls[i], c=colors[j], marker=markers[i], label=label)
            j += 1
        i += 1
    plt.title('AWGN Channel')
    plt.xlabel('k/n')
    plt.ylabel(metric)
    if metric == 'psnr':
        plt.ylim(0, 35)
    else:
        plt.ylim(0.4, 1)
    plt.grid(True)
    plt.legend(loc='lower right')
    os.makedirs('./plot/plot1_{0}'.format(metric), exist_ok=True)
    plt.savefig('./plot/plot1_{0}/{1}_CompRatio{2}_SNR{3}.png'.format(metric, model_str, compression_ratios, snr_train))
    plt.show()

def all_model_compare(metric, model_str, snr_train):
    colors = list(mcolors.TABLEAU_COLORS)
    i = 0
    for snr in snr_train:
        j = 0
        for model in model_str:
            path = './result_txt/plot1/{0}_SNR{1}.txt'.format(model, snr)
            with open(path, 'r') as f:
                text = f.read()
                compression_ratios = text.split('\n')[0]
                compression_ratios = json.loads(compression_ratios)
                if metric == 'psnr':
                    psnr = text.split('\n')[1]
                    metric_score = json.loads(psnr)
                else:
                    ssim = text.split('\n')[2]
                    metric_score = json.loads(ssim)
            label = '{0} (SNR={1}dB)'.format(model, snr)
            plt.plot(compression_ratios, metric_score, ls='-', c=colors[j], marker='o', label=label)
            j += 1
        i += 1
        plt.title('AWGN Channel')
        plt.xlabel('k/n')
        plt.ylabel(metric)
        if metric == 'psnr':
            plt.ylim(0, 35)
        else:
            plt.ylim(0.4, 1)
        plt.grid(True)
        plt.legend(loc='lower right')
        os.makedirs('./plot/plot1_{0}'.format(metric), exist_ok=True)
        plt.savefig('./plot/plot1_{0}/{1}_CompRatio{2}_SNR{3}.png'.format(metric, model_str, compression_ratios, snr_train))
        plt.show()

def test_plot(model_str, snr_train, compression_ratios):
    for comp_ratio in compression_ratios:
        colors = list(mcolors.TABLEAU_COLORS)
        markers = ['o', 's']
        i = 0
        ls = ['--', '-']
        for model in model_str:
            j = 0
            for snr in snr_train:
                path = './result_txt/plot2/{0}_CompRatio{1}_SNR{2}.txt'.format(model, comp_ratio, snr)
                with open(path, 'r') as f:
                    text = f.read()
                    snr_test = text.split('\n')[0]
                    snr_test = json.loads(snr_test)
                    psnr = text.split('\n')[1]
                    psnr = json.loads(psnr)
                label = '{0} (SNR={1}dB)'.format(model, snr)
                plt.plot(snr_test, psnr, ls=ls[i], c=colors[j], marker=markers[i], label=label)
                j += 1
            i += 1
        plt.title('AWGN Channel (k/n={0})'.format(comp_ratio))
        plt.xlabel('SNR_test (dB)')
        plt.ylabel('PSNR (dB)')
        plt.ylim(0,35)
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.savefig('./plot/plot2/{0}_CompRatio{1}_SNR{2}.png'.format(model_str, comp_ratio, snr_train))
        plt.show()

def all_img(model):
    fig = plt.figure()
    i=1
    for filename in sorted(glob.glob('./img/{0}/*.jpg'.format(model))):
        #print(filename)
        img = cv2.imread(filename)
        ax = fig.add_subplot(3,3,i)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

        label = filename.replace('./img/{0}/pred_'.format(model), '')
        label = label.replace('.jpg', '')
        print(label)
        ax.set_xlabel(label)
        i += 1
    plt.savefig('./img/{0}/all.jpg'.format(model))
    plt.show()

### RUN ###
model = 'model12'
model_f = model12
snr_train = [0,10,20]
snr_test = [2, 4, 7, 10, 13, 16, 18, 22, 25, 27] #[2, 10, 18, 26]
compression_ratios = [0.06, 0.26, 0.49] #0.06, 0.26, 0.49

### TRAIN & EVALAUATION ###
#train(model, model_f, snr_train, compression_ratios, x_train, x_test, batch_size=100, epochs=20)
#comp_eval(model, x_test, testX, compression_ratios, snr_train)
#test_eval(model, x_test, testX, compression_ratios, snr_train, snr_test)

### VISUALIZATION ###
all_img('basic')
model_list = ['basic', model]
#comp_plot('psnr', model_list, snr_train)
#comp_plot('ssim', model_list, snr_train)
#test_plot(model_list, snr_train, compression_ratios)

### compare many models ###
#model_list = ['basic', model]
#all_model_compare('psnr', model_list, snr_train)
#all_model_compare('ssim', model_list, snr_train)