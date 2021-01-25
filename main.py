import load_mnist
import Tree
import matplotlib.pyplot as plt

def display_image(pixels):
    img = pixels.reshape([28,28])
    plt.imshow(img, cmap=plt.get_cmap('gray_r'))
    plt.show()

if __name__ == '__main__':

    N_TRAIN = 30
    images, labels = load_mnist.load_mnist('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
    s_images = images[:N_TRAIN]
    s_labels = labels[:N_TRAIN]
    tree = Tree.Tree(s_images, s_labels)

    print(" predict: ", tree.predict(images[N_TRAIN+1]))
    print(" actual: ", labels[N_TRAIN+1])

    while True:
        which = int(input("Which img do you want to classify?: "))
        print(" predict: ", tree.predict(images[which]))
        print(" actual: ", labels[which])
