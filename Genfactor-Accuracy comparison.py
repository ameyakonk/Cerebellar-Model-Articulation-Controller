import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
from Discrete_CMAC import Discrete
from Continuous_CMAC import Continuous

if __name__ == "__main__":
    x = np.linspace(0, 2*np.pi, 100)
    #y  = np.absolute(np.sin(x))
    y = np.cos(x)

    data = list(zip(x, y))
    random.shuffle(data)
    train_data = data[:70]
    test_data = data[70:]

    weights = 34
    gen = 1
    min_ = 0
    max_ = 2*np.pi

    discrete_obj = Discrete(weights, gen)
    continuous_obj = Continuous(weights, gen)
    
    discrete_accuracy = []
    continous_accuracy = []
    Timer_Dis = []
    Timer_Cont = []

    for i in range(weights):
        weight_vec,time_ = discrete_obj.train(i, weights, train_data, min_, max_)
        disc_accuracy = discrete_obj.predict(test_data, min_, max_, weight_vec, True)
        print(time_)
        discrete_accuracy.append(disc_accuracy)
        Timer_Dis.append(time_)
       
        weight_vec, time_ = continuous_obj.train(i, weights, train_data, min_, max_)
        cont_accuracy = continuous_obj.predict(test_data, min_, max_, weight_vec, True)
        continous_accuracy.append(cont_accuracy)
        Timer_Cont.append(time_)
       

    x = np.linspace(1, weights, weights)
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(x, discrete_accuracy, label = "Discrete CMAC")
    axs[0].plot(x, continous_accuracy, 'g', label = "Continuous CMAC")
    axs[0].set_xlabel("gen factor")
    axs[0].set_ylabel("accuracy")

    axs[1].plot(x, Timer_Dis, label = "Discrete CMAC")
    axs[1].set_xlabel("gen factor")
    axs[1].set_ylabel("time")

    axs[2].plot(x, Timer_Cont, label = "Continuous CMAC")
    axs[2].set_xlabel("gen factor")
    axs[2].set_ylabel("time")

    axs[2].legend()
    axs[1].legend()
    axs[0].legend()
    plt.show()
    


