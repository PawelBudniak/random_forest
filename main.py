import load_mnist
import Tree
import matplotlib.pyplot as plt
import numpy as np
import random_forest
import test
from collections import Counter


def display_image(pixels):
    img = pixels.reshape([28, 28])
    plt.imshow(img, cmap=plt.get_cmap('gray_r'))
    plt.show()


if __name__ == '__main__':

    TRAIN_IMG_PATH, TRAIN_LABEL_PATH = 'data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte'
    TEST_IMG_PATH, TEST_LABEL_PATH = 'data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte'

    START = 0
    N_TRAIN = 6000
    STOP = START + N_TRAIN
    N_TEST = N_TRAIN // 6
    images, labels = load_mnist.load_mnist(TRAIN_IMG_PATH, TRAIN_LABEL_PATH)
    s_images = images[START:STOP]
    s_labels = labels[START:STOP]
    # test_images, test_labels = load_mnist.load_mnist(TEST_IMG_PATH, TEST_LABEL_PATH)
    test_images = images[STOP: STOP + N_TEST]
    test_labels = labels[STOP: STOP + N_TEST]

    # test.k_fold_validation(s_images, s_labels, 4, random_forest.Forest)

    # tree = random_forest.Forest(s_images, s_labels, n_trees=20, training_size=0.8, n_features=1.0,
    #                             split_method='choose_best', thresholds=[10, 30, 50], max_features='sqrt',
    #                             min_feature_entropy=0.001)

    model, error = test.k_fold_model_and_error(s_images, s_labels, 4, random_forest.Forest,
                                               n_trees=20, training_size=0.8, n_features=1.0,
                                               split_method='thresholds', thresholds=[50],
                                               max_features='sqrt', min_feature_entropy=0.05
                                               )
    print('k-fold error: ', error)


    # tree = Tree.Tree(data=s_images, labels=s_labels, split_method='thresholds', thresholds=[50], max_features=None, min_feature_entropy=None)

    # print('k-fold: ', test.k_fold_validation(s_images, s_labels, k=5, model_constructor=random_forest.Forest, n_trees=10,
    #                                          training_size=0.8))

    print("train error:", test.error_rate(images[START: STOP], labels[START: STOP], model))
    print("test error: ", test.error_rate(test_images, test_labels, model))

    while True:
        which = int(input("Which img do you want to classify?: "))
        print(" predict: ", model.predict(images[which]))
        print(" actual: ", labels[which])
        display_image(images[which])
