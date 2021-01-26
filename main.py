import load_mnist
import Tree
import matplotlib.pyplot as plt
import numpy as np
import random_forest
import test


def display_image(pixels):
    img = pixels.reshape([28, 28])
    plt.imshow(img, cmap=plt.get_cmap('gray_r'))
    plt.show()


if __name__ == '__main__':

    TRAIN_IMG_PATH, TRAIN_LABEL_PATH = 'data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte'
    TEST_IMG_PATH, TEST_LABEL_PATH = 'data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte'


    START = 0
    N_TRAIN = 20000
    STOP = START + N_TRAIN
    N_TEST = N_TRAIN // 6
    images, labels = load_mnist.load_mnist(TRAIN_IMG_PATH, TRAIN_LABEL_PATH)
    s_images = images[START:STOP]
    s_labels = labels[START:STOP]
    test_images, test_labels = load_mnist.load_mnist(TEST_IMG_PATH, TEST_LABEL_PATH)

    # test.k_fold_validation(s_images, s_labels, 4, random_forest.Forest)

    tree = random_forest.Forest(s_images, s_labels, n_trees=50, training_size=0.8, n_features=0.9, split_method='thresholds', thresholds=[50])

    # tree = Tree.Tree(data=s_images, labels=s_labels, split_method='thresholds', thresholds=[20, 60])

    #
    # # s_images[s_images < 50] = 0
    # # s_images[s_images >= 50] = 1
    # #
    # #
    # # np.savetxt('imgs.csv', s_images.astype(int), fmt='%i', delimiter=',')
    # # np.savetxt('labels.csv', s_labels.astype(int), fmt='%i', delimiter=',')

    # print('k-fold: ', test.k_fold_validation(s_images, s_labels, k=5, model_constructor=random_forest.Forest, n_trees=10,
    #                                          training_size=0.8))

    # train_correct = 0
    # for i in range(START, STOP):
    #     if tree.predict(images[i]) == labels[i]:
    #         train_correct += 1
    # test_correct = 0
    # for i in range(STOP, N_TEST + STOP):
    #     if tree.predict(images[i]) == labels[i]:
    #         test_correct += 1
    #
    print("train error:", test.error_rate(images[START: STOP], labels[START: STOP], tree))
    print("test error: ", test.error_rate(test_images, test_labels, tree))

    while True:
        which = int(input("Which img do you want to classify?: "))
        print(" predict: ", tree.predict(images[which], verbose=True))
        print(" actual: ", labels[which])
        display_image(images[which])
