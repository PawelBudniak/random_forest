import load_mnist
import Tree
import matplotlib.pyplot as plt
import numpy as np


def display_image(pixels):
    img = pixels.reshape([28, 28])
    plt.imshow(img, cmap=plt.get_cmap('gray_r'))
    plt.show()


if __name__ == '__main__':

    START = 0
    N_TRAIN = 10000
    STOP = START+N_TRAIN
    N_TEST = N_TRAIN // 6
    images, labels = load_mnist.load_mnist('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
    s_images = images[START:STOP]
    s_labels = labels[START:STOP]
    tree = Tree.Tree(s_images, s_labels)

    print(" predict: ", tree.predict(images[START + 1]))
    print(" actual: ", labels[START + 1])

    train_correct = 0
    for i in range(START, STOP):
        if tree.predict(images[i]) == labels[i]:
            train_correct += 1
    test_correct = 0
    for i in range(STOP, N_TEST + STOP):
        if tree.predict(images[i]) == labels[i]:
            test_correct += 1

    # s_images[s_images < 50] = 0
    # s_images[s_images >= 50] = 1
    #
    #
    # np.savetxt('imgs.csv', s_images.astype(int), fmt='%i', delimiter=',')
    # np.savetxt('labels.csv', s_labels.astype(int), fmt='%i', delimiter=',')

    print("train error:", 1 - train_correct / N_TRAIN)
    print("test error: ", 1 - test_correct / N_TEST)

    while True:
        which = int(input("Which img do you want to classify?: "))
        print(" predict: ", tree.predict(images[which]))
        print(" actual: ", labels[which])
        display_image(images[which])
