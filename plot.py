import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import glob
import cv2

from model import *


def comp_plot(metric, model_str, snr_train):
    if metric == 'psnr':
        colors = list(mcolors.TABLEAU_COLORS)
    else:
        colors = list(mcolors.BASE_COLORS)

    markers = ['o', 's', '^']
    ls = ['-', '--', ':']
    i = 0
    for model in model_str:
        j = 0
        for snr in snr_train:
            path = './result_txt/plot1/{0}/SNR{1}.txt'.format(model, snr)
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
        plt.ylim(0, 30)
    else:
        plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1,0.7))
    plt.tight_layout()
    os.makedirs('./plot/plot1_{0}'.format(metric), exist_ok=True)
    plt.savefig('./plot/plot1_{0}/{1}_CompRatio{2}_SNR{3}.png'.format(metric, model_str, compression_ratios, snr_train))
    plt.show()

def test_plot(metric, model_str, snr_train, compression_ratios):
    for comp_ratio in compression_ratios:
        if metric == 'psnr':
            colors = list(mcolors.TABLEAU_COLORS)
        else:
            colors = list(mcolors.BASE_COLORS)
            #colors = colors[3:]
        markers = ['o', 's', '^']
        i = 0
        ls = ['-', '--', ':']
        for model in model_str:
            j = 0
            for snr in snr_train:
                path = './result_txt/plot2/{0}/CompRatio{1}_SNR{2}.txt'.format(model, comp_ratio, snr)
                with open(path, 'r') as f:
                    text = f.read()
                    snr_test = text.split('\n')[0]
                    snr_test = json.loads(snr_test)
                    if metric == 'psnr':
                        psnr = text.split('\n')[1]
                        metric_score = json.loads(psnr)
                    else:
                        ssim = text.split('\n')[2]
                        metric_score = json.loads(ssim)
                label = '{0} (SNR={1}dB)'.format(model, snr)
                plt.plot(snr_test, metric_score, ls=ls[i], c=colors[j], marker=markers[i], label=label)
                j += 1
            i += 1
        plt.title('AWGN Channel (k/n={0})'.format(comp_ratio))
        plt.xlabel('SNR_test (dB)')
        plt.ylabel(metric)
        if metric == 'psnr':
            plt.ylim(10, 30)
        else:
            plt.ylim(0, 1)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1,0.7))
        plt.tight_layout()
        os.makedirs('./plot/plot2_{0}'.format(metric), exist_ok=True)
        plt.savefig(
            './plot/plot2_{0}/{1}_CompRatio{2}_SNR{3}.png'.format(metric, model_str, compression_ratios, snr_train))
        plt.show()

def all_model_compare(metric, model_str, snr_train):
    colors = list(mcolors.TABLEAU_COLORS)
    i = 0
    for snr in snr_train:
        j = 0
        for model in model_str:
            path = './result_txt/plot1/{0}/SNR{1}.txt'.format(model, snr)
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
            if snr == 0:
                plt.ylim(10,18)
            elif snr == 10:
                plt.ylim(10,25)
            else:
                plt.ylim(10, 35)
        else:
            if snr == 0:
                plt.ylim(0,1)
            elif snr == 10:
                plt.ylim(0.4,1)
            else:
                plt.ylim(0.6,1)
        plt.grid(True)
        plt.legend(loc='lower right')
        os.makedirs('./plot/plot1_{0}'.format(metric), exist_ok=True)
        plt.savefig('./plot/plot1_{0}/{1}_CompRatio{2}_SNR{3}.png'.format(metric, model_str, compression_ratios, snr_train))
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

