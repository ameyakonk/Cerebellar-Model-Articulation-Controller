import numpy as np
import math
import time

class Discrete():
    def __init__(self, weights, gen):
        self.weights = weights
        self.m_mat = []
        self.association_mat = {}
    
    def s_m_mapping(self, min_, max_, input_val) :
        if input_val < min_:
            return 1
        elif input_val > max_:
            return self.weights - self.gen 
        
        #print(int(math.floor((self.weights - self.gen + 1)*(input_val-min_)/(max_ - min_) + 1)))
        return int(math.floor((self.weights - self.gen - 1)*(input_val-min_)/(max_ - min_) + 1))
    
    def m_matrix(self, data, min_, max_):
        self.m_mat_size = self.weights - self.gen + 1
        for data_val in data:
            self.association_mat[data_val[0]] = self.s_m_mapping( min_, max_, data_val[0]) 

    def train(self, gen, weights, data, min_, max_):
        self.gen  = gen
        weight_vec = np.ones(weights)
        self.association_mat = {}

        self.m_matrix(data, min_, max_)

        count = 0
        max_count  = 10000
        isReached = False
        alpha = 0.1
        threshold = 0.0001
        prev_error = 0
        cur_error = 0
        start = time.time()
        while count < max_count and not isReached:
            prev_error = cur_error
            for data_val in data:
                start = self.association_mat[data_val[0]]
                y_predicted = np.sum(weight_vec[start: start + self.gen])
                error = data_val[1] - y_predicted
                correction = alpha*error/self.gen
                weight_vec[start: start + self.gen] = weight_vec[start: start + self.gen] + correction
             
            accuracy =  self.predict(data, min_, max_, weight_vec, False) 
           
            cur_error = 1 - accuracy
            if(abs(cur_error - prev_error) < threshold):
                isReached = True 
            count += 1
        #print(weight_vec)
        end  = time.time()
        
        return weight_vec, end - start

    def predict(self, data, min_, max_, weight_vec, cond):
        y_predicted = []
        y_actual = []
        x_actual = []
        if cond == True:
          self.m_matrix(data, min_, max_)
        for data_val in data:
            start = self.association_mat[data_val[0]]
            y_ = np.sum(weight_vec[start: start + self.gen])
            y_predicted.append(y_)
            y_actual.append(data_val[1])
            x_actual.append(data_val[0]) 

        return self.calculate_accuracy(y_predicted, y_actual)
        
    def calculate_accuracy(self, ypred, yact):
        error = np.subtract(ypred, yact)
        error_sq = np.power(error,2)
        sum_error = np.sum(error_sq)
        error = math.sqrt(sum_error)/len(ypred)
        return 1 - error
    
    def output_finder(self, x, weight_vec):
        y_ = []
        for data_ in x:
            start = self.association_mat[data_]
            y_.append(np.sum(weight_vec[start: start + self.gen]))
        return y_
