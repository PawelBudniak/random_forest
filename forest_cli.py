import random_forest
import sys
import load_mnist
import main
import test



def get_forest_from_args(data, data_labels, args):
    kwargs = {}
    i = 0
    while i != len(args):
        if args[i] == '-n_trees':
            kwargs['n_trees'] = int(args[i+1])
            i += 1
        if args[i] == '-training_size':
            kwargs['training_size'] = float(args[i+1])
            i += 1
        if args[i] == '-n_features':
            kwargs['n_features'] = float(args[i+1])
            i += 1
        if args[i] == '-split_method':
            kwargs['split_method'] = args[i+1]
            i += 1
        if args[i] == '-thresholds':
            if args[i+1] == 'None':
                kwargs['thresholds'] = None
            else:
                # handle list
                thresholds = []
                i += 1
                while not args[i].startswith('-') and i != len(args):
                    thresholds.append(float(args[i]))
                    i += 1
                kwargs['thresholds'] = thresholds
                continue  # i is already at next argument
        if args[i] == '-max_features':
            if args[i+1] == 'None':
                kwargs['max_features'] = None
            else:
                kwargs['max_features'] = args[i+1]
            i += 1
            pass
        if args[i] == '-min_feature_entropy':
            if args[i + 1] == 'None':
                kwargs['min_feature_entropy'] = None
            else:
                kwargs['min_feature_entropy'] = float(args[i + 1])
            i += 1

        i += 1

    print(kwargs)
    return random_forest.Forest(data, data_labels, **kwargs)



if __name__ == '__main__':

    images, data_labels = load_mnist.load_mnist(main.TRAIN_IMG_PATH, main.TRAIN_LABEL_PATH)
    test_images, test_labels = load_mnist.load_mnist(main.TEST_IMG_PATH, main.TEST_LABEL_PATH)

    model = get_forest_from_args(images, data_labels, sys.argv)

    print("test error: ", test.error_rate(test_images, test_labels, model))

    user_input = input("Which img do you want to classify? (q to quit): ")
    while user_input != 'q':
        which = int(user_input)
        print(" predict: ", model.predict(test_images[which]))
        print(" actual: ", test_labels[which])
        main.display_image(test_images[which])
        user_input = input("Which img do you want to classify? (q to quit): ")
