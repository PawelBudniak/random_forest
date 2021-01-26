import numpy as np
import math
import Tree
from collections import Counter


class Forest:
    def __init__(self, data, labels, *, n_trees=10, training_size=1.0, n_features=1.0):

        if isinstance(n_features, float):
            n_features = int(n_features * data.shape[1])
        if isinstance(training_size, float):
            training_size = int(training_size * data.shape[0])

        self.trees = []
        for i in range(n_trees):
            # generate random training subset with replacement
            training_ids = np.random.choice(data.shape[0], size=training_size, replace=True)
            training_set = data[training_ids]

            # generate random feature subset without replacement
            # n_features = math.floor(math.sqrt(squared_n_features))
            feature_ids = sorted(np.random.choice(data.shape[1], size=n_features, replace=False))
            # training_set = training_set[:, feature_ids]
            labels_subset = labels[training_ids]

            new_tree = Tree.Tree(data=training_set, labels=labels_subset, features=set(feature_ids))
            self.trees.append(new_tree)

    def predict(self, input, verbose=False):
        votes = [tree.predict(input) for tree in self.trees]
        if verbose:
            print(Counter(votes))
        return Counter(votes).most_common(1)[0][0]
