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
    N_TEST = N_TRAIN // 6
    images, labels = load_mnist.load_mnist(TRAIN_IMG_PATH, TRAIN_LABEL_PATH)
    TOTAL_LEN = len(labels)
    s_images = images[START:STOP]
    s_labels = labels[START:STOP]
    # test_images, test_labels = load_mnist.load_mnist(TEST_IMG_PATH, TEST_LABEL_PATH)
    test_images = images[STOP: STOP + N_TEST]
    test_labels = labels[STOP: STOP + N_TEST]

    ## TODO: zmien to
    test_param_name = 'n_features'

    fname = 'results_' + test_param_name + '.csv'
    columns = [test_param_name, 'czas trenowania', 'sr. czas predykcji', 'blad testowy']

    with open(fname, 'w') as fp:
        fp.write(','.join(columns) + '\n')

    # tree = random_forest.Forest(s_images, s_labels, n_trees=20, training_size=0.8, n_features=1.0,
    #                             split_method='choose_best', thresholds=[10, 30, 50], max_features='sqrt',
    #                             min_feature_entropy=0.001)

    ## TODO: zmien to
    N_LOOP = 5
    test_param_value = 1.0
    param_step = 0.05

    # poradnik:
    # training_size 1.0 -> 0.1, step = 0.1
    # n_features 1.0 -> 0.5, step = 0.05
    # min_feature_entropy = None, pozniej (0.001, 0.101), step = 0.01
    # max_features = {None, 'sqrt'} -- none zajmie duzo chyba
    # c45 = True

    # n_trees 1->5->10,20,30,40,50
    # split_method = 'choose_best' i thresholds jakies wymysle xD

    for i in range(N_LOOP):

        # TODO: ustaw tu
        model, error, train_time = test.k_fold_model_and_error(s_images, s_labels, 3, random_forest.Forest,
                                                               training_size=test_param_value)

        test_param_value -= param_step

        print('k-fold error: ', error)

        pred_start = perf_counter_ns()
        for _ in range(N_PRED):
            model.predict(images[random.randint(0, TOTAL_LEN)])
        pred_end = perf_counter_ns()

        avg_pred_time = (pred_end - pred_start) / N_PRED
        avg_pred_time = avg_pred_time / test.NS_TO_MS

        with open(fname, 'a') as fp:
            line = map(str, [test_param_value, train_time, avg_pred_time, error])
            fp.write(','.join(line) + '\n')

        print("train error:", test.error_rate(images[START: STOP], labels[START: STOP], model))
        print("test error: ", test.error_rate(test_images, test_labels, model))

    # while True:
    #     which = int(input("Which img do you want to classify?: "))
    #     print(" predict: ", model.predict(images[which]))
    #     print(" actual: ", labels[which])
    #     display_image(images[which])

    # model, error, train_time = test.k_fold_model_and_error(s_images, s_labels, 4, random_forest.Forest,
    #                                                        n_trees=10, training_size=test_param_value, n_features=1.0,
    #                                                        split_method='thresholds', thresholds=[50],
    #                                                        max_features='sqrt', min_feature_entropy=0.05
    #                                                        )
