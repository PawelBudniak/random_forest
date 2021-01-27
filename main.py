'''
File used for testing the models performance and error rate, meant to run in IDE, for cli use forset_cli.py
'''

import load_mnist
import matplotlib.pyplot as plt
import numpy as np
import random_forest
import test
from time import perf_counter_ns
import random


def display_image(pixels):
    img = pixels.reshape([28, 28])
    plt.imshow(img, cmap=plt.get_cmap('gray_r'))
    plt.show()


TRAIN_IMG_PATH = 'data/train-images.idx3-ubyte'
TRAIN_LABEL_PATH = 'data/train-labels.idx1-ubyte'
TEST_IMG_PATH, TEST_LABEL_PATH = 'data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte'

if __name__ == '__main__':

    N_PRED = 100
    START = 0
    N_TRAIN = 5000
    STOP = START + N_TRAIN
    images, labels = load_mnist.load_mnist(TRAIN_IMG_PATH, TRAIN_LABEL_PATH)
    TOTAL_LEN = len(labels)
    s_images = images[START:STOP]
    s_labels = labels[START:STOP]

    test_param_name = 'n_features'

    fname = 'results_' + test_param_name + '.csv'
    columns = [test_param_name, 'czas trenowania', 'sr. czas predykcji', 'blad testowy']

    with open(fname, 'w') as fp:
        fp.write(','.join(columns) + '\n')


    N_LOOP = 5
    N_REPEAT = 20
    test_param_value = 1.0
    param_step = -0.05
    K_FOLD = 3


    for i in range(N_LOOP):

        av_pred_time = 0
        av_error = 0
        av_train_time = 0

        for i in range(N_REPEAT):

            model, error, train_time = test.k_fold_model_and_error(s_images, s_labels, K_FOLD, random_forest.Forest,
                                                                   n_features=test_param_value)

            # tree = RandomForestClassifier()
            # tree.fit(s_images, s_labels)
            av_error += error / N_REPEAT
            av_train_time += train_time / N_REPEAT
            print('k-fold error: ', error)

            pred_start = perf_counter_ns()
            for _ in range(N_PRED):
                model.predict(images[random.randint(0, TOTAL_LEN - 1)])
            pred_end = perf_counter_ns()

            pred_time = (pred_end - pred_start) / N_PRED
            pred_time = pred_time / test.NS_TO_MS

            av_pred_time += pred_time / N_REPEAT

        with open(fname, 'a') as fp:
            line = map(str, [test_param_value, av_train_time, av_pred_time, av_error])
            fp.write(','.join(line) + '\n')

        test_param_value += param_step
