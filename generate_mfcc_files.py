import time

import numpy as np
import python_speech_features as sf
import os
import scipy.io.wavfile as wav


def fwav2mfcc(infilename, outfilename):
    fs, signal = wav.read(infilename)
    wav_feature = sf.mfcc(signal, fs)
    d_mfcc_feat = sf.base.delta(wav_feature, 1)
    d_mfcc_feat2 = sf.base.delta(wav_feature, 2)
    feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2)).T
    (dim, frame_no) = feature.shape
    sampSize = dim * 4
    parmKind = 838
    np.savez(outfilename, frame_no=frame_no, sampSize=sampSize, parmKind=parmKind,
             feature=feature)


def gengrate_mfcc_samples(indir='wav', in_filter='\.[Ww][Aa][Vv]', outdir='mfcc', out_ext='.npz', outfile_format='htk',
                          frame_size_sec=0.025, frame_shift_sec=0.010, use_hamming=1, pre_emp=0, bank_no=26,
                          cep_order=12, lifter=22, delta_win_weight=np.ones((1, 2 * 2 + 1))):
    # print(delta_win_weight)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for root, dirs, files in os.walk(indir):
        for file in files:
            inputfilename = os.path.join(root, file)
            outdir_ = outdir + '\\' + root.split('\\')[-1]
            outfilename = outdir_ + '\\' + file.split('.')[0] + out_ext
            # print(outdir_)
            # print(outfilename)
            if not os.path.exists(outdir_):
                os.makedirs(outdir_)
            fwav2mfcc(inputfilename, outfilename, outfile_format, frame_size_sec, frame_shift_sec, use_hamming, pre_emp,
                      bank_no, cep_order, lifter, delta_win_weight)


def generate_testing_list(list_filename='testingfile_list.npz'):
    MODEL_NO = 11
    dir1 = 'mfcc'
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
    dir1 = 'mfcc'
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


def generate_mfcc_file():
    start = time.time()
    gengrate_mfcc_samples()
    generate_training_list()
    generate_testing_list()
    print(time.time()-start)


if __name__ == '__main__':
    generate_mfcc_file()
