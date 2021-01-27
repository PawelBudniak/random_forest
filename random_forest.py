import numpy as np
import math
import Tree
from collections import Counter


class Forest:
    def __init__(self, data, labels, *, n_trees=20, training_size=1.0, n_features=1.0,
                 split_method='thresholds', thresholds=None, max_features='sqrt', min_feature_entropy=0.001, c45=False):
        """
        A random forest classifier, grows n_trees and predicts labels by having each of the trees vote on their
        classification.

        Mandatory args:
        :param data: training data
        :param labels: data labels

        Optional kwargs:
        :param n_trees: int
                number of trees in the forest
        :param training_size: int or float
                number of training samples randomly chosen for each tree with replacement
                if int - flat number
                if float - number of samples = data.shape[0] * training_size
        :param n_features: float
                relative number of features randomly chosen without replacement used to train each tree
                calculated after filtering with min_feature_entropy!
        :param split_method: {'mean','thresholds', 'choose_best'}
                How to split the data.
                if 'mean' the split point is the mean of the feature's values
                if 'thresholds' chosen, splits are made on all thresholds points
                if 'choose_best', the best binary split is chosen from thresholds points
        :param thresholds: 1D array-like(float or int)
                thresholds used for 'thresholds' and 'choose_best' split methods
                if split_method is 'thresholds' or 'choose_best', and thresholds is None,
                it's set to a one elemnent list [50]
        :param max_features: {'sqrt'}
                if 'sqrt' -  max_features = sqrt(n_features)
                if None   -  max_features = n_features
                consider only up to max_features when searching for the best split
                except if no valid split has been found, the search continues
        :param min_feature_entropy: float
                for each feature column it's entropy is calculated
                if it's below or equal to this threshold, that feature is discarded
                if None - no filtering is done
        :param c45: bool
                if true, apply c45 pruning on the trees
        """

        if thresholds is None and split_method in ('thresholds', 'choose_best'):
            thresholds = [50]

        if isinstance(training_size, float):
            training_size = int(training_size * data.shape[0])

        if max_features is not None:
            features = set(range(data.shape[1]))
            features = list(Tree.filter_features(data, features, min_feature_entropy))

        # flat n_features values can't be allowed, only float values are accepted
        # because features are filtered and we don't know up front what their final size will be
        if not isinstance(n_features, float):
            raise ValueError('n_features parameter must be float')
        n_features = int(n_features * len(features))

        self.trees = []
        for i in range(n_trees):
            # generate random training subset with replacement
            training_ids = np.random.choice(data.shape[0], size=training_size, replace=True)
            training_set = data[training_ids]

            # generate random feature subset without replacement
            # n_features = math.floor(math.sqrt(squared_n_features))
            feature_ids = np.random.choice(features, size=n_features, replace=False)
            # training_set = training_set[:, feature_ids]
            labels_subset = labels[training_ids]

            new_tree = Tree.Tree(data=training_set, labels=labels_subset, features=set(feature_ids),
                                 split_method=split_method, thresholds=thresholds, max_features=max_features,
                                 min_feature_entropy=min_feature_entropy, c45=c45)
            self.trees.append(new_tree)

    def predict(self, input, verbose=False):
        votes = [tree.predict(input) for tree in self.trees]
        if verbose:
            print(Counter(votes))
        return Counter(votes).most_common(1)[0][0]
