import random
import numpy as np
import numpy as np
from neuron import neuron
import random
from matplotlib import pyplot as plt
from recep_field import rf
import cv2
from spike_train import encode
from rl import rl
from rl import update
from reconstruct import reconst_weights
from parameters import param as par
from var_th import threshold
import time


neuron2class={1:0,2:1,0:-1}
accuracy = []


if __name__ == '__main__':



    synapse=np.load("synapse.npy")
    # time series
    time = np.arange(1, par.T + 1, 1)
    layer_2 = []
    for i in range(par.n):
        a = neuron()
        layer_2.append(a)
#再加一层for循环 k（0-9）代表神经元的序号（也就是真正的类的值） k_i i-- 0-900? 总共9000张
    for i in range(1, 11):
        # print i,"  ",k
        img = cv2.imread("neuron" + str(i) + ".bmp", 0)
        # img = cv2.imread(str(k)+"_" + str(i) + ".bmp", 0)
        # Convolving image with receptive field
        pot = rf(img)

        # Generating spike train
        train = np.array(encode(pot))

        # calculating threshold value for the image
        var_threshold = threshold(train)

        # print var_threshold
        # synapse_act = np.zeros((par.n,par.m))
        # var_threshold = 9
        # print var_threshold
        # var_D = (var_threshold*3)*0.07

        # 漏电值
        var_D = 0.15 * par.scale

        for x in layer_2:
            x.initial(var_threshold)

        # flag for lateral inhibition
        f_spike = 0



        active_pot = []
        for index1 in range(par.n):
            active_pot.append(0)

        # Leaky integrate and fire neuron dynamics
        for t in time:
            for j, x in enumerate(layer_2):
                active = []
                if (x.t_rest < t):
                    x.P = x.P + np.dot(synapse[j], train[:, t])
                    if (x.P > par.Prest):
                        x.P -= var_D
                    active_pot[j] = x.P

                # pot_arrays[j].append(x.P)

            # Lateral Inhibition
            if (f_spike == 0):
                high_pot = max(active_pot)
                if (high_pot > var_threshold):
                    f_spike = 1
                    winner = np.argmax(active_pot)
                    #i的范围：eg. 1-900
                   # accuracy[f] = neuron2class[winner]==k
                    #  f预设0   f=f+1
                    #sum(accuracy)/总数 得出准确率？
                    print("the winner is " + str(winner))