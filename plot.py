import json
import glob
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    os.makedirs('./plot/plot1_{0}}'.format(metric), exist_ok=True)
    plt.savefig('./plot/plot1_{0}/{1}_CompRatio{2}_SNR{3}.png'.format(metric, model_str, compression_ratios, snr_train))
    plt.show()

'''def comp_ssim_plot(model_str, snr_train):
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
                ssim = text.split('\n')[2]
                ssim = json.loads(ssim)
            label = '{0} (SNR={1}dB)'.format(model, snr)
            plt.plot(compression_ratios, ssim, ls=ls[i], c=colors[j], marker=markers[i], label=label)
            #plt.plot(compression_ratios, ssim, ls='-', c=colors[i], marker='X', label=label)
            j += 1
        i += 1
    plt.title('AWGN Channel')
    plt.xlabel('k/n')
    plt.ylabel('SSIM')
    plt.ylim(0.4,1)
    plt.grid(True)
    plt.legend(loc='lower right')
    os.makedirs('./plot/plot1_ssim', exist_ok=True)
    plt.savefig('./plot/plot1_ssim/{0}_CompRatio{1}_SNR{2}.png'.format(model_str, compression_ratios, snr_train))
    plt.show()'''

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