import numpy as np
import time
import random
import math
from Discrete_CMAC import Discrete

class Continuous:
    def __init__(self, weights, gen):
        self.weights = weights
        self.gen = gen
        self.m_mat = []
        self.association_mat = {}
    
    def s_m_mapping(self, min_, max_, input_val) :
        if input_val < min_:
            return 1
        elif input_val > max_:
            return self.weights - self.gen 
        
        #print(int(math.floor((self.weights - self.gen + 1)*(input_val-min_)/(max_ - min_) + 1)))
        return (self.weights - self.gen - 1)*(input_val-min_)/(max_ - min_) + 1
    
    def m_matrix(self, data, min_, max_):
        self.m_mat_size = self.weights - self.gen + 1
        for data_val in data:
            left_ = int(np.floor(self.s_m_mapping( min_, max_, data_val[0])))
            left_ = self.m_mat_size-1 if left_ > self.m_mat_size else left_
            right_ = int(np.ceil(self.s_m_mapping( min_, max_, data_val[0])))
            right_ = self.m_mat_size-1 if right_ > self.m_mat_size else right_
            
            if left_ != right_: 
                self.association_mat[data_val[0]] = (left_, right_)
            else: 
                self.association_mat[data_val[0]] = (left_, 0)
            
    def train(self, gen, weights, data, min_, max_):
        self.gen  = gen
        weight_vec = np.ones(weights)
        self.association_mat = {}
        input_X_ = np.linspace(min_, max_, self.weights + 1 - self.gen)
        self.m_matrix(data, min_, max_)
    
        count = 0
        max_count  = 10000
        isReached = False
        alpha = 0.01
        threshold = 0.0001
        prev_error = 0
        cur_error = 0
        start = time.time()
        while count < max_count and not isReached:
            prev_error = cur_error
            for data_val in data:
                asc_wt_ind_flr = self.association_mat[data_val[0]][0]
                asc_wt_ind_ceil = self.association_mat[data_val[0]][1]
                
                if asc_wt_ind_flr >= len(input_X_):
                    asc_wt_ind_flr = len(input_X_)-1

                if asc_wt_ind_ceil >= len(input_X_):
                    asc_wt_ind_ceil = len(input_X_)-1
                
                left_common = np.abs(input_X_[asc_wt_ind_flr] - data_val[0])
                right_common = np.abs(input_X_[asc_wt_ind_ceil] - data_val[0])
                
                left_contri_ratio = right_common/(left_common + right_common)
                right_contri_ratio = left_common/(left_common + right_common)
                
                y_output = (left_contri_ratio * np.sum(weight_vec[asc_wt_ind_flr: asc_wt_ind_flr + self.gen])) + (right_contri_ratio * np.sum(weight_vec[asc_wt_ind_ceil: asc_wt_ind_ceil + self.gen]))
                
                error = data_val[1] - y_output
                correction = alpha*error/self.gen
                
                weight_vec[asc_wt_ind_flr: (asc_wt_ind_flr + self.gen)] = \
                                            [(weight_vec[ind] + correction) for ind in range(asc_wt_ind_flr, (asc_wt_ind_flr + self.gen))]
                
                weight_vec[asc_wt_ind_ceil: (asc_wt_ind_ceil + self.gen)] = \
                                            [(weight_vec[ind] + correction) for ind in range(asc_wt_ind_ceil, (asc_wt_ind_ceil + self.gen))]

                # weight_vec[asc_wt_ind_flr: asc_wt_ind_flr + self.gen] = weight_vec[asc_wt_ind_flr: asc_wt_ind_flr + self.gen] + correction
                # weight_vec[asc_wt_ind_ceil: asc_wt_ind_ceil + self.gen] = weight_vec[asc_wt_ind_ceil: asc_wt_ind_ceil + self.gen] + correction

            accuracy =  self.predict(data, min_, max_, weight_vec, False) 
            end  = time.time()
            cur_error = 1 - accuracy
            if(abs(cur_error - prev_error) < threshold):
                isReached = True 
            count += 1
        return weight_vec, end-start

    def output_finder(self, x, weight_vec, min_, max_):
        y_ = []
        input_X_vec = np.linspace(min_, max_, self.weights + 1 - self.gen)

        for test_val in x:
            asc_wt_ind_flr = self.association_mat[test_val][0]
            asc_wt_ind_ceil = self.association_mat[test_val][1]

            if asc_wt_ind_flr >= len(input_X_vec):
                    asc_wt_ind_flr = len(input_X_vec)-1

            if asc_wt_ind_ceil >= len(input_X_vec):
                    asc_wt_ind_ceil = len(input_X_vec)-1
            left_common = np.abs(input_X_vec[asc_wt_ind_flr] - test_val)
            right_common = np.abs(input_X_vec[asc_wt_ind_ceil] - test_val)

            left_contri_ratio = right_common/(left_common + right_common)
            right_contri_ratio = left_common/(left_common + right_common)
            y_output = (left_contri_ratio * np.sum(weight_vec[asc_wt_ind_flr: asc_wt_ind_flr + self.gen])) + (right_contri_ratio * np.sum(weight_vec[asc_wt_ind_ceil: asc_wt_ind_ceil + self.gen]))
            y_.append(y_output)
        return y_

    def calculate_accuracy(self, ypred, yact):
        error = np.subtract(ypred, yact)
        error_sq = np.power(error,2)
        sum_error = np.sum(error_sq)
        error = math.sqrt(sum_error)/len(ypred)
        return error

    def predict(self, data, min_, max_, weight_vec, cond):
        output = []
        y_actual = []
        input_X_vec = np.linspace(min_, max_, self.weights + 1 - self.gen)

        if cond:
            self.m_matrix(data, min_, max_)
        
        for ind, test_val in enumerate(data):
            asc_wt_ind_flr = self.association_mat[test_val[0]][0]
            asc_wt_ind_ceil = self.association_mat[test_val[0]][1]

            if asc_wt_ind_flr >= len(input_X_vec):
                    asc_wt_ind_flr = len(input_X_vec)-1

            if asc_wt_ind_ceil >= len(input_X_vec):
                    asc_wt_ind_ceil = len(input_X_vec)-1
            left_common = np.abs(input_X_vec[asc_wt_ind_flr] - test_val[0])
            right_common = np.abs(input_X_vec[asc_wt_ind_ceil] - test_val[0])

            left_contri_ratio = right_common/(left_common + right_common)
            right_contri_ratio = left_common/(left_common + right_common)
            y_actual.append(test_val[1])
            y_output = (left_contri_ratio * np.sum(weight_vec[asc_wt_ind_flr: asc_wt_ind_flr + self.gen])) + (right_contri_ratio * np.sum(weight_vec[asc_wt_ind_ceil: asc_wt_ind_ceil + self.gen]))
            output.append(y_output)
        
        error = self.calculate_accuracy(output, y_actual)
        accuracy = 1 - error
        return accuracy
                   