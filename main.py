import time

import numpy as np
from model import HMM
from generate_mfcc_files import generate_mfcc_file


def main():
    # generate_mfcc_file()
    training_file_list_name = 'trainingfile_list.npz'
    testing_file_list_name = 'testingfile_list.npz'
    DIM = 39
    num_of_model = 11
    num_of_state_start = 12
    num_of_state_end = 15
    accuracy_rate = np.zeros(num_of_state_end + 1)
    hmm = HMM()
    for i in range(num_of_state_start, num_of_state_end + 1):
        print('开始训练')
        start = time.time()
        hmm.EM_HMMtraining(training_file_list_name, DIM, num_of_model, i)
        accuracy_rate[i] = hmm.HMMtesting(testing_file_list_name)
        end = time.time()
        res = 'num_of_state: {}, accuracy_rate: {}, time: {}'\
            .format(i, accuracy_rate[i], end - start)
        print(res)
        print()


if __name__ == '__main__':
    main()
