import concurrent.futures
from tqdm import tqdm
import numpy as np

np.seterr(divide='ignore')


class HMM:
    def __init__(self):
        pass

    # 初始化模型
    def EM_initialization_model(self, training_file_list_name, DIM, num_of_state, num_of_model):
        sum_of_features = np.zeros((DIM, 1))
        sum_of_features_square = np.zeros((DIM, 1))
        num_of_feature = 0
        training_file_list = np.load(training_file_list_name)['training_list']
        num_of_uter = len(training_file_list)

        for i in range(num_of_uter):
            filename = training_file_list[i, 1]
            file_now = np.load(filename)
            features = file_now['feature']

            sum_of_features = sum_of_features + np.reshape(np.sum(features, 1).T, (-1, 1))
            sum_of_features_square = sum_of_features_square + np.reshape(np.sum(np.power(features, 2), 1).T, (-1, 1))
            num_of_feature = num_of_feature + np.size(features, 1)
            '''
            if i==10:
                print(np.sum(features, 1))
                print(sum_of_features.shape)
                print(sum_of_features_square.shape)
                print(sum_of_features)
                print(num_of_feature)
            '''
        self.calculate_inital_EM_HMM_items(num_of_state, num_of_model, sum_of_features, sum_of_features_square,
                                           num_of_feature)

    # 初始话均值，方差以及aij
    def calculate_inital_EM_HMM_items(self, num_of_state, num_of_model, sum_of_features, sum_of_features_square,
                                      num_of_feature):
        for k in range(num_of_model):
            for m in range(num_of_state):
                self.mean[:, m, k] = np.reshape(sum_of_features / num_of_feature, -1)
                self.var[:, m, k] = np.reshape(sum_of_features_square / num_of_feature - np.reshape(
                    np.multiply(self.mean[:, m, k], self.mean[:, m, k]), (-1, 1)), -1)

            for i in range(1, num_of_state + 1):
                self.Aij[i, i + 1, k] = 0.4
                self.Aij[i, i, k] = 1 - self.Aij[i, i + 1, k]

            self.Aij[0, 1, k] = 1

        # print(self.Aij[:,:,1])

    def logGaussian(self, mean_i, var_i, o_i):
        dim = len(var_i)
        log_b = -0.5 * (dim * np.log(2 * np.pi) + np.sum(np.log(var_i)) + np.sum(
            np.divide(np.multiply(o_i - mean_i, o_i - mean_i), var_i)))
        return log_b

    def log_sum_alpha(self, log_alpha_t, aij_j):
        len_x = len(log_alpha_t)
        y = np.full([len_x], -np.inf)
        ymax = -np.inf
        for i in range(len_x):
            y[i] = log_alpha_t[i] + np.log(aij_j[i])
            if y[i] > ymax:
                ymax = y[i]

        if ymax == np.inf:
            logsumalpha = np.inf
        else:
            sum_exp = 0
            for i in range(len_x):
                if ymax == -np.inf and y[i] == -np.inf:
                    sum_exp = sum_exp + 1
                else:
                    sum_exp = sum_exp + np.exp(y[i] - ymax)
            logsumalpha = ymax + np.log(sum_exp)
        return logsumalpha

    def log_sum_beta(self, aij_i, mean, var, obs, beta_t1):
        aij_i = np.reshape(aij_i, -1)
        beta_t1 = np.reshape(beta_t1, -1)
        obs = np.reshape(obs, -1)

        temp, len_x = mean.shape
        y = np.full([len_x], -np.inf)
        ymax = -np.inf
        for j in range(len_x):
            y[j] = np.log(aij_i[j]) + self.logGaussian(mean[:, j], var[:, j], obs) + beta_t1[j]
            if y[j] > ymax:
                ymax = y[j]
        if ymax == np.inf:
            logsumbeta = np.inf
        else:
            sum_exp = 0
            for i in range(len_x):
                if ymax == -np.inf and y[i] == -np.inf:
                    sum_exp = sum_exp + 1
                else:
                    sum_exp = sum_exp + np.exp(y[i] - ymax)
            logsumbeta = ymax + np.log(sum_exp)
        return logsumbeta

    # ER迭代
    def EM_HMM_FR(self, mean, var, aij, obs, k=None):
        # ----------------------------
        file_now = np.load(obs)
        obs = file_now['feature']
        dim, T = obs.shape

        mean = np.concatenate((np.full([dim, 1], np.nan), mean, np.full([dim, 1], np.nan)), axis=1)
        var = np.concatenate((np.full([dim, 1], np.nan), var, np.full([dim, 1], np.nan)), axis=1)
        aij[-1, -1] = 1
        temp, N = mean.shape
        log_alpha = np.full([N, T + 1], -np.inf)
        log_beta = np.full([N, T + 1], -np.inf)

        for i in range(N):
            log_alpha[i, 0] = np.log(aij[0, i]) + self.logGaussian(mean[:, i], var[:, i], obs[:, 0])

        for t in range(1, T):
            for j in range(1, N - 1):
                log_alpha[j, t] = self.log_sum_alpha(log_alpha[1:N - 1, t - 1], aij[1:N - 1, j]) + self.logGaussian(
                    mean[:, j], var[:, j], obs[:, t])
        log_alpha[N - 1, T] = self.log_sum_alpha(log_alpha[1:N - 1, T - 1], aij[1:N - 1, N - 1])

        log_beta[:, T - 1] = np.log(aij[:, N - 1])
        for t in range(T - 2, -1, -1):
            for i in range(1, N - 1):
                log_beta[i, t] = self.log_sum_beta(aij[i, 1:N - 1], mean[:, 1:N - 1], var[:, 1:N - 1], obs[:, t + 1],
                                                   log_beta[1:N - 1, t + 1])
        log_beta[N - 1, 0] = self.log_sum_beta(aij[0, 1:N - 1], mean[:, 1:N - 1], var[:, 1:N - 1], obs[:, 0],
                                               log_beta[1:N - 1, 0])

        log_Xi = np.full([N, N, T], -np.inf)
        for t in range(T - 1):
            for j in range(1, N - 1):
                for i in range(1, N - 1):
                    log_Xi[i, j, t] = log_alpha[i, t] + np.log(aij[i, j]) + self.logGaussian(mean[:, j],
                                                                                             var[:, j], obs[:, t + 1]) + \
                                      log_beta[j, t + 1] - log_alpha[N - 1, T]
        for i in range(N):
            log_Xi[i, N - 1, T - 1] = log_alpha[i, T - 1] + np.log(aij[i, N - 1]) - log_alpha[N - 1, T]

        log_gamma = np.full([N, T], -np.inf)
        for t in range(T):
            for i in range(1, N - 1):
                log_gamma[i, t] = log_alpha[i, t] + log_beta[i, t] - log_alpha[N - 1, T]
        gamma = np.exp(log_gamma)

        mean_numerator = np.zeros((dim, N))
        var_numerator = np.zeros((dim, N))
        denominator = np.zeros((N, 1))
        aij_numerator = np.zeros((N, N))

        for j in range(1, N - 1):
            for t in range(0, T):
                mean_numerator[:, j] = mean_numerator[:, j] + gamma[j, t] * obs[:, t]
                var_numerator[:, j] = var_numerator[:, j] + gamma[j, t] * np.multiply(obs[:, t], obs[:, t])
                denominator[j] = denominator[j] + gamma[j, t]

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for t in range(T):
                    aij_numerator[i, j] = aij_numerator[i, j] + np.exp(log_Xi[i, j, t])

        log_likelihood = log_alpha[N - 1, T]
        likelihood = np.exp(log_alpha[N - 1, T])

        return mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood, likelihood, k

    # 模型训练
    def EM_HMMtraining(self, training_file_list_name='trainingfile_list.npz', DIM=39, num_of_model=11, num_of_state=12):
        self.mean = np.zeros((DIM, num_of_state, num_of_model))
        self.var = np.zeros((DIM, num_of_state, num_of_model))
        self.Aij = np.zeros((num_of_state + 2, num_of_state + 2, num_of_model))
        self.EM_initialization_model(training_file_list_name, DIM, num_of_state, num_of_model)
        num_of_iteration = 2
        log_likelihood_iter = np.zeros(num_of_iteration)
        likelihood_iter = np.zeros(num_of_iteration)
        training_file_list = np.load(training_file_list_name)['training_list']
        # print(training_file_list)
        num_of_uter = len(training_file_list)

        for iter in range(num_of_iteration):
            print('    第' + str(iter + 1) + '次迭代')
            # 重新设置变量
            sum_mean_numerator = np.zeros((DIM, num_of_state, num_of_model))
            sum_var_numerator = np.zeros((DIM, num_of_state, num_of_model))
            sum_aij_numerator = np.zeros((num_of_state, num_of_state, num_of_model))
            sum_denominator = np.zeros((num_of_state, num_of_model))
            log_likelihood = 0
            likelihood = 0

            k_arr = [int(file[0]) - 1 for file in training_file_list]
            filename_arr = training_file_list[:, 1]
            mean_arr = [self.mean[:, :, k] for k in k_arr]
            var_arr = [self.var[:, :, k] for k in k_arr]
            Aij_arr = [self.Aij[:, :, k] for k in k_arr]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                for res in tqdm(executor.map(self.EM_HMM_FR, mean_arr, var_arr, Aij_arr, filename_arr, k_arr),
                                total=len(k_arr), desc=f'epoch {iter}'):
                    mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood_i, likelihood_i, k = res
                    sum_mean_numerator[:, :, k] += mean_numerator[:, 1:-1]
                    sum_var_numerator[:, :, k] += var_numerator[:, 1:-1]
                    sum_aij_numerator[:, :, k] += aij_numerator[1:-1, 1:-1]
                    sum_denominator[:, k] += np.reshape(denominator[1:-1], -1)

                    log_likelihood += log_likelihood_i
                    likelihood += likelihood_i

            for k in range(num_of_model):
                for n in range(num_of_state):
                    self.mean[:, n, k] = sum_mean_numerator[:, n, k] / sum_denominator[n, k]
                    self.var[:, n, k] = sum_var_numerator[:, n, k] / sum_denominator[n, k] - np.multiply(
                        self.mean[:, n, k], self.mean[:, n, k])

            for k in range(num_of_model):
                for i in range(1, num_of_state + 1):
                    for j in range(1, num_of_state + 1):
                        self.Aij[i, j, k] = sum_aij_numerator[i - 1, j - 1, k] / sum_denominator[i - 1, k]
                self.Aij[num_of_state, num_of_state + 1, k] = 1 - self.Aij[num_of_state, num_of_state, k]

            self.Aij[num_of_state + 1, num_of_state + 1, num_of_model - 1] = 1
            log_likelihood_iter[iter] = log_likelihood
            likelihood_iter[iter] = likelihood

    def HMMtesting(self, testing_file_list_name='testingfile_list.npz'):
        print('开始测试')
        num_of_model = 11
        num_of_error = 0
        testing_file_list = np.load(testing_file_list_name)['testing_list']
        num_of_uter, temp = testing_file_list.shape
        num_of_testing = num_of_uter
        filename_arr = testing_file_list[:, 1]
        k_arr = [int(file[0]) - 1 for file in testing_file_list]
        number_of_model_arr = [num_of_model for file in testing_file_list]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in tqdm(executor.map(self._test, filename_arr, k_arr, number_of_model_arr), total=len(k_arr), desc='test'):
                num_of_error += res

        accuracy_rate = (num_of_testing - num_of_error) * 100 / num_of_testing
        return accuracy_rate

    def _test(self,filename, k, num_of_model):
        file_now = np.load(filename)
        features = file_now['feature']
        fopt_max = -np.inf
        digit = -1
        num_of_error = 0
        for p in range(num_of_model):
            fopt = self.viterbi_dist_FR(self.mean[:, :, p].copy(), self.var[:, :, p].copy(),
                                        self.Aij[:, :, p].copy(), features)
            if fopt > fopt_max:
                digit = p
                fopt_max = fopt
        if digit != k:
            num_of_error = num_of_error + 1
        return num_of_error

    def viterbi_dist_FR(self, mean, var, aij, obs):
        dim, t_len = obs.shape
        mean = np.concatenate((np.full([dim, 1], np.nan), mean, np.full([dim, 1], np.nan)), axis=1)
        var = np.concatenate((np.full([dim, 1], np.nan), var, np.full([dim, 1], np.nan)), axis=1)
        aij[-1, -1] = 1
        temp, m_len = mean.shape
        fjt = np.full([m_len, t_len], -np.inf)
        for j in range(1, m_len - 1):
            fjt[j, 0] = np.log(aij[0, j]) + self.logGaussian(mean[:, j], var[:, j], obs[:, 0])

        for t in range(1, t_len):
            for j in range(1, m_len - 1):
                f_max = -np.inf
                i_max = -1
                f = -np.inf
                for i in range(1, j + 1):
                    if fjt[i, t - 1] > -np.inf:
                        f = fjt[i, t - 1] + np.log(aij[i, j]) + self.logGaussian(mean[:, j], var[:, j], obs[:, t])
                    if f > f_max:
                        f_max = f
                        i_max = i
                if i_max != -1:
                    fjt[j, t] = f_max

        fopt = -np.inf
        for i in range(1, m_len - 1):
            f = fjt[i, t_len - 1] + np.log(aij[i, m_len - 1])
            if f > fopt:
                fopt = f
        return fopt
