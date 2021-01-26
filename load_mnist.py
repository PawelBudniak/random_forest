import numpy as np


# for large files
def read_binary(path):
    # number of bytes read from a file with each fp.read
    CHUNK_SIZE = 4096

    with open(path, 'rb') as fp:
        chunk = fp.read(CHUNK_SIZE)
        while chunk:
            for b in chunk:
                yield b
            chunk = fp.read(CHUNK_SIZE)


def load_mnist(images_path, labels_path, byte_order='big'):
    with open(images_path, 'rb') as fp:
        images = fp.read()

    OFFSET = 16

    n_images = int.from_bytes(images[4:8], byte_order)
    n_rows = int.from_bytes(images[8:12], byte_order)
    n_cols = int.from_bytes(images[12:OFFSET], byte_order)

    # print(n_images, n_rows, n_cols)

    array_cols = n_rows * n_cols
    images_array = np.empty(shape=(n_images, array_cols), dtype=np.ubyte)

    for i in range(n_images):
        # list() to convert byte string to list of ints in range 0-255
        images_array[i] = list(images[OFFSET + i * array_cols: OFFSET + (i + 1) * array_cols])

    # read labels
    with open(labels_path, 'rb') as fp:
        labels = fp.read()

    n_labels = n_images
    L_OFFSET = 8
    labels_array = np.array(list(labels[L_OFFSET: n_labels + L_OFFSET]))

    return images_array, labels_array
