import math
import os
import time
import wave

import scipy.fftpack
from pyforest import sns
from scipy.fftpack import fft, ifft
import numpy as np
from matplotlib import pyplot as plt

path = '语音.wav'
f = 44100
alpha = 0.97
# Frame length
wlen = 0
# Frame duration
wlen_time = 0.025
# Frame movement
inc = 0
# Frame movement duration
inc_time = 0.010
# number of mel filter
mel_filter_number = 26
# Maximum detectable frequency
max_f = 8000
# Minimum detectable frequency
min_f = 300


def save_features(data, path):
    np.savez(path, feature=data)


def draw_wav(data, title, xlabel='sample number', ylabel="range"):
    """
    Draw the (time---sampling point diagram) of audio
    :param data: Audio data
    :param title: title of diagram
    :param xlabel: name of xlable
    :param ylabel: name of ylable
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    # Set horizontal axis label
    plt.xlabel(xlabel)
    # Set vertical axis label
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(title + ".jpg")
    plt.show()


def draw_frequency(data, title, xlabel='sample number', ylabel="range"):
    """
    Draw the diagram with relation of Frequency and Magnitude in certain frame
    :param data: Audio data
    :param title: title of diagram
    :param xlabel: name of xlable
    :param ylabel: name of ylable
    """
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, len(data)) * f / wlen, data)
    # Set horizontal axis label
    plt.xlabel(xlabel)
    # Set vertical axis label
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(title + ".jpg")
    plt.show()


def draw_thermodynamic_diagram(data, title, xlabel='', ylabel='', vmax=''):
    """
    Draw thermal diagram
    :param data:Audio data
    :param title: title of diagram
    :param xlabel: name of xlable
    :param ylabel: name of ylable
    :return:
    """
    plt.figure(figsize=(15, 10))
    sns.set()
    # cmap is a parameter of the color of the heat map
    if vmax == '':
        ax = sns.heatmap(data.T, cmap="rainbow", xticklabels=100, yticklabels=6).invert_yaxis()
    else:
        vmax = int(vmax)
        ax = sns.heatmap(data.T, cmap="rainbow", xticklabels=100, yticklabels=100, vmax=vmax).invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(title + ".jpg")
    plt.show()


def pre_emphasis(data):
    """
    pre-emphasis

    Pre-emphasis increases the magnitude of higher frequencies in the speech signal compared with lower frequencies

    x^′[t_d−1]=x[t_d]−alpha * x[t_d−1]
    :param data: Audio data
    :return: audio data after pre_emphasis
    """
    new_data = [0] * len(data)
    for i in range(1, len(data)):
        new_data[i - 1] = data[i] - alpha * data[i - 1]
    new_data[-1] = data[-1]
    return new_data


def digital_to_analog_conversion():
    """
    Read a wav file data
    """
    global f
    with wave.open(path) as fi:
        params = fi.getparams()
        print(params)
        # Read audio frames
        nframes = fi.getnframes()
        # Read sampling frequency
        framerate = fi.getframerate()
        # Read all frames
        frames = fi.readframes(nframes)
        # Save as an array
        framerates = np.frombuffer(frames, dtype=np.short)
        times = fi.getnframes() * 1000 / framerate
        global wlen
        global inc
        wlen = int(wlen_time * framerate)
        inc = int(inc_time * framerate)
        f = framerate

    return framerates


def frame(wave_data):
    """
    Framing
    extracting the time points of all frames to obtain the matrix of nf * wlen length
    :param wave_data: Raw data of sound
    :return: For the data after framing, frames [0] represents the first frame, and so on
    """
    signal_length = len(wave_data)
    # If the signal length is less than the length of one frame, the number of frames is defined as 1
    if signal_length <= wlen:
        nf = 1
    # Otherwise, the total length of the frame is calculated
    else:
        nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))
    # The total flattened length of all frames
    pad_length = int((nf - 1) * inc + wlen)
    # The insufficient length is filled with 0, which is similar to the expansion array operation in FFT
    zeros = np.zeros((pad_length - signal_length,))
    # The filled signal is recorded as pad_signal
    pad_signal = np.concatenate((wave_data, zeros))
    #  is equivalent to extracting the time points of all frames to obtain the matrix of nf * wlen length
    """
    The format of indices is as follows:
    first line：0 1 ... wlen-1
    second line：inc inc+1 ... inc+wlen-1
    ...
    inc'th line:...
    """
    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
    # Convert indices to matrix
    indices = np.array(indices, dtype=np.int32)
    # Get frame signal
    frames = pad_signal[indices]

    return frames


def hamming_windowing(data):
    """
    hamming windowing

    the direct truncation of the signal (with rectangular window) will produce spectrum leakage, in order to
    improve the spectrum leakage, add non rectangular window, generally add Hamming window, because the amplitude
    frequency characteristic of Hamming window is that the sidelobe attenuation is large

    w[n]=(1−α)−α*cos(2πn/L−1) a=0.46164
    :param data: Audio data
    :return: Windowed audio data
    """
    w = [0] * wlen
    for i in range(wlen - 1):
        w[i] = (1 - 0.46164) - 0.46164 * math.cos((2 * math.pi * i) / (wlen - 1))
    w = np.array(w)
    for i in range(len(data)):
        data[i] = w * data[i]
    return data


def wave_fft(data):
    """
    fast Fourier transform
    extracts spectral information from a windowed signal
    :param data: Audio data
    :return: Frequency energy relationship per frame
    """
    fft_y = fft(data)
    fft_y = np.abs(fft_y)
    return fft_y


def mel_filter_function(x, fm_pre, fm, fm_next):
    """
    Mel filter evaluation
    :param x: independent variable
    :param fm_pre: f(m-1)
    :param fm: Current f(m)
    :param fm_next: f(m+1)
    :return: Strain variable
    """
    # Convert frequency bin to frequency
    x = x * f / wlen
    if x < fm_pre or x > fm_next:
        return 0
    elif x == fm:
        return 1
    elif fm_pre < x < fm:
        return (x - fm_pre) / (fm - fm_pre)
    elif fm < x < fm_next:
        return (fm_next - x) / (fm_next - fm)
    else:
        return 0


def mel_filter(wave_data):
    """
    mel filter
    Apply a mel-scale filter bank to DFT power spectrum to obtain mel-scale power spectrum
    f_mel(f)=1127ln(1+f/700)
    Y_t[m]=∑(H_m[k]|X_t[k]|^2)
    :param wave_data: raw data
    :return: data after mel filter
    """
    global max_f
    max_f = f / 2
    # Divide f_mel(300)--f_mel(max_f) into 26 parts on average
    dots = np.linspace(1127 * math.log(min_f / 700 + 1), 1127 * math.log(max_f / 700 + 1), mel_filter_number + 2)
    # convert it return to frequency
    for i in range(len(dots)):
        dots[i] = np.floor(700 * (math.exp(dots[i] / 1127) - 1))
    res = np.zeros((len(wave_data), mel_filter_number), dtype=float)
    H = np.array([0] * len(wave_data[0]), dtype=float)
    for i in range(len(res)):
        # 26 filters
        for j in range(1, len(dots) - 1):
            for k in range(len(wave_data[i])):
                H[k] = mel_filter_function(k, dots[j - 1], dots[j], dots[j + 1])
            res[i][j - 1] = np.sum(H * wave_data[i])
            # Prevent log (0)
            if res[i][j - 1] <= math.pow(2, -10):
                res[i][j - 1] = math.pow(2, -10)
    return res


def DCT(data):
    """
    DCT
    Noise removal
    c_t[j]=∑(log(Y_t[m])*cos((m+0.5)jπ/M))    j=0,…,C−1
    :param data: fbank_features, data [i] represents the fbank features of frame i
    :return: data after DCT
    """
    res = np.zeros((len(data), mel_filter_number), dtype=float)
    for i in range(len(res)):
        for j in range(len(res[i])):
            haha = []
            # get cos array
            for m in range(mel_filter_number):
                haha.append(np.cos((m + 0.5) * j * np.pi / mel_filter_number))
            res[i][j] = np.sum(data[i] * np.array(haha))
    res = np.array(res)
    return res


def get_energy(data):
    """
    Find the energy per frame
    energy[i]=log(∑(x[i]*x[i]))
    :param data: data after windowing
    :return: the energy per frame
    """
    energy = [0] * len(data)
    energy = np.array(energy, dtype=float)
    for i in range(len(data)):
        # Log is added so that the order of energy is the same as that of fbank features
        all = np.sum(data[i] * data[i], dtype=float)
        if all <= math.pow(2, -10):
            all = math.pow(2, -10)
        energy[i] = np.log(all)
        # energy[i] = energy[i] / f
    return energy


def dynamic_features(data, energy):
    """
    Dynamic feature extraction
    Speech is not constant frame-to-frame, so we can add features to do with how the cepstral coefficients change over time
    :param data: data after DCT, data [i] represents the characteristics of frame i
    :param energy: Energy per frame
    :return: Results of dynamic feature extraction
    """
    res = np.zeros((len(data), 39), dtype=float)
    for i in range(len(res)):
        for j in range(len(res[i])):
            if j < 12:
                res[i][j] = data[i][j + 1]
            # get energy
            elif j == 12:
                res[i][j] = energy[i]
            elif i == 0 or i == len(res) - 1:
                res[i][j] = data[i][j % 13 + 1]
            # First order difference
            elif 12 < j < 27:
                res[i][j] = (data[i + 1][j % 13 + 1] - data[i - 1][j % 13 + 1]) / 2

    for i in range(len(res)):
        for j in range(len(res[i])):
            if i == 0 or i == len(res) - 1:
                continue
            # Second order difference
            elif 26 < j < 40:
                res[i][j] = (res[i + 1][j - 13] - res[i - 1][j - 13]) / 2
    res = np.array(res)
    return res


def feature_normalization(data):
    """
    feature normalization
    Divide feature vector by standard deviation of feature vectors, so each feature vector element has a variance of 1
    :param data: raw data, data [i] represents the characteristics of frame i
    :return: Normalized data
    """
    # We normalize the same feature of each frame, so we transpose the original data first
    data = data.T
    for i in range(len(data)):
        data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])
    data = data.T
    return data


def generate_one_mfcc(inpath, outpath):
    global path
    path = inpath
    # data fetch
    wav_data = digital_to_analog_conversion()
    # pre-emphasis
    wav_data = pre_emphasis(wav_data)
    # framing
    wav_data = frame(wav_data)
    # hamming windowing
    wav_data = hamming_windowing(wav_data)
    # Find the energy of each frame
    energy = get_energy(wav_data)
    # fast Fourier transform
    wav_data = wave_fft(wav_data)
    # mel_filter
    fbank_features = np.log(mel_filter(wav_data))
    # DCT
    features = DCT(fbank_features)
    # Dynamic feature extraction
    res = dynamic_features(features, energy)
    # feature normalization
    # res = feature_normalization(res)
    # Draw thermal diagram
    # draw_thermodynamic_diagram(res, 'dynamic features after normalization', 'frame', '')
    # Save final results
    save_features(res.T, outpath)


def generate_all_mfcc(indir, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for root, dirs, files in os.walk(indir):
        for file in files:
            inputfilename = os.path.join(root, file)
            outdir_ = outdir + '\\' + root.split('\\')[-1]
            outfilename = outdir_ + '\\' + file.split('.')[0] + '.npz'
            # print(inputfilename)
            # print(outfilename)
            if not os.path.exists(outdir_):
                os.makedirs(outdir_)
            generate_one_mfcc(inputfilename, outfilename)


def generate_testing_list(list_filename='testingfile_list.npz'):
    MODEL_NO = 11
    dir1 = 'mymfcc'
    dir3 = ['AH', 'AR', 'AT', 'BC', 'BE', 'BM', 'BN', 'CC', 'CE', 'CP', 'DF', 'DJ', 'ED', 'EF', 'ET', 'FA', 'FG',
            'FH', 'FM', 'FP', 'FR', 'FS', 'FT', 'GA', 'GP', 'GS', 'GW', 'HC', 'HJ', 'HM', 'HR', 'IA', 'IB', 'IM',
            'IP', 'JA', 'JH', 'KA', 'KE', 'KG', 'LE', 'LG', 'MI', 'NL', 'NP', 'NT', 'PC', 'PG', 'PH', 'PR', 'RK',
            'SA', 'SL', 'SR', 'SW', 'TC']
    wordids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    testingfile = []
    for dir in dir3:
        num = 0
        for item in wordids:
            num = num + 1
            for ch in ['A', 'B']:
                s = '{0}/{1}/{2}{3}_endpt.npz'.format(dir1, dir, item, ch)
                testingfile.append([num, s])
    np.savez(list_filename, testing_list=testingfile)


def generate_training_list(list_filename='trainingfile_list.npz'):
    MODEL_NO = 11
    dir1 = 'mymfcc'
    dir3 = ['AE', 'AJ', 'AL', 'AW', 'BD', 'CB', 'CF', 'CR', 'DL', 'DN', 'EH', 'EL', 'FC', 'FD', 'FF', 'FI', 'FJ',
            'FK', 'FL', 'GG']
    wordids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    trainingfile = []
    for dir in dir3:
        num = 0
        for item in wordids:
            num = num + 1
            for ch in ['A', 'B']:
                s = '{0}/{1}/{2}{3}_endpt.npz'.format(dir1, dir, item, ch)
                trainingfile.append([num, s])
    np.savez(list_filename, training_list=trainingfile)


def generate_my_mfcc_file():
    start = time.time()
    generate_all_mfcc('wav', 'mymfcc')
    generate_training_list()
    generate_testing_list()
    print(time.time() - start)


if __name__ == '__main__':
    generate_my_mfcc_file()
