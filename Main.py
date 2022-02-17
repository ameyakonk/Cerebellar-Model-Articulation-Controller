import matplotlib.pyplot as plt
import numpy as np
import time
import random
import sys
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

    # x_train = list(list(zip(*train_data))[0])
    # y_train = list(list(zip(*train_data))[1])

    # x_test = list(list(zip(*test_data))[0])
    # y_test = list(list(zip(*test_data))[1])

    weights = 35
    gen = 10
    min_ = 0
    max_ = 2*np.pi

    discrete_obj = Discrete(weights, gen)
    weight_vec,_ = discrete_obj.train(gen, weights, train_data, min_, max_)
    accuracy = discrete_obj.predict(test_data, min_, max_, weight_vec, True)
    print("Discrete Accuracy", end = " ")
    print(accuracy)

    continuous_obj = Continuous(weights, gen)
    weight_vec,_ = continuous_obj.train(gen, weights, train_data, min_, max_)
    accuracy = continuous_obj.predict(test_data, min_, max_, weight_vec, True)
    print("Continuous Accuracy", end = " ")
    print(accuracy)

    y_new_dis = discrete_obj.output_finder(x,weight_vec)
    y_new_cont = continuous_obj.output_finder(x,weight_vec, min_, max_)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, y)
    axs[0].plot(x, y_new_dis, 'r', label = "Discrete CMAC")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("cos(x)")
    axs[1].plot(x, y_new_cont, 'g', label = "Continuous CMAC")
    axs[1].plot(x, y)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("cos(x)")
    
    axs[0].legend()
    axs[1].legend()
    
    plt.show()
    
