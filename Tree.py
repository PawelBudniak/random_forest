import numpy as np
from collections import Counter
import math
import statistics
import random


def entropy(counts):
    total = sum(counts)
    return sum(-count / total * math.log(count / total) for count in counts)


def sequence_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = math.e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def remove_useless_features(data, features, epsilon=None):

    if epsilon is None:
        # if there's only one possible value for a feature it gives no information
        # we can safely ignore it
        return set(filter(lambda f: len(set(data[:, f])) != 1, features))
    # if epsilon specified, remove features with entropy less than epsilon
    return set(filter(lambda f: sequence_entropy(data[:, f]) > epsilon, features))


class Tree:
    SPLIT_METHODS = {'mean', 'thresholds', 'choose_best'}

    def __init__(self, data, labels, *, features=None, split_method='mean', thresholds=None, max_features=None,
                 min_feature_entropy=None):
        """
              Mandatory args:
              :param data: training data
              :param labels: data labels

              Optional kwargs:
              :param features: default=None
                     data features, if not specified they are computed as: set(range(data.shape[1]))
              :param split_method: {'mean','thresholds'}, default = 'mean'
                     how to split the nodes, if 'thresholds' chosen, the thresholds must be set in param thresholds
              :param thresholds: Iterable(float or int): default=None
                     thresholds used to split the nodes
              :param max_features: {'sqrt'}, default=None
                     if 'sqrt' -  max_features = sqrt(n_features)
                     if None   -  max_features = n_features
                     consider only up to max_features when searching for the best split
                     except if no valid split has been found, then the search continues

              """

        if features is None:
            features = set(range(data.shape[1]))  # each column represents one pixel - one feature

        if min_feature_entropy is not None:
            features = remove_useless_features(data, features, epsilon=min_feature_entropy)

        if split_method not in self.SPLIT_METHODS:
            raise ValueError('Possible split methods: ', self.SPLIT_METHODS)
        self.split_method = split_method

        if split_method in ('thresholds', 'choose_best'):
            if thresholds is None:
                raise RuntimeError(f'split_method=\'{split_method}\' chosen without specifying thresholds parameter')
            self.thresholds = sorted(thresholds)
            # last threshold must be math inf
            if self.thresholds[-1] != math.inf:
                self.thresholds.append(math.inf)

        if max_features == 'sqrt':
            self.max_features = math.sqrt(len(features))
        elif max_features is None:
            self.max_features = len(features)


        self.root = self.id3(labels, features, data)

    def best_split(self, labels, feature, data, split_method='mean', data_entropy=None):
        """
        :param labels: data labels
        :param feature: feature to find a split on
        :param data: training data
        :param split_method: {'mean','thresholds', 'choose_best'}, default = 'mean'
               How to split the data.
                if 'mean' the split point is the mean of the feature's values
                if 'thresholds' chosen, splits are made on all self.thresholds points
                if 'choose_best', the best binary split is chosen from self.thresholds points
        :param data_entropy: pre-calculated entropy of the data labels


        :return: np.array(float): thresholds, last threshold is always math.inf
                ,np.array(np.array): indices of data split on thresholds
                ,float: information gain of the split


        """
        inf_gain = None

        if split_method == 'mean':
            threshold = statistics.mean(data[:, feature])

            idx_data_less = np.where(data[:, feature] < threshold)[0]
            idx_data_gte = np.where(data[:, feature] >= threshold)[0]

            # last threshold must be math inf
            thresholds = np.array([threshold, math.inf])
            split = np.array([idx_data_less, idx_data_gte])

        if split_method == 'thresholds':
            thresholds = np.array(self.thresholds)
            split = []
            for i, value in enumerate(thresholds):
                prev_value = thresholds[i - 1] if i != 0 else -math.inf
                indices = np.where((value > data[:, feature]) & (data[:, feature] >= prev_value))[0]
                split.append(indices)
            split = np.array(split)

        if split_method == 'choose_best':
            if self.thresholds is None:
                raise RuntimeError('No specified thresholds')

            thresholds, split, inf_gain = self.best_bin_split(labels, feature, data, data_entropy)

        # find non-empty subsets
        non_empty_ids = [i for i in range(len(split)) if len(split[i]) > 0]
        split = split[non_empty_ids]
        thresholds = thresholds[non_empty_ids]

        # if there is only one non-empty subset, the threshold to enter it is math.inf
        if len(thresholds) == 1:
            thresholds[0] = math.inf

        # if inf_gain wasn't calculated earlier by choose_best split method, calculate it here
        if inf_gain is None:
            inf_gain = self.split_inf_gain(data, labels, split, data_entropy)

        return thresholds, split, inf_gain

    def best_bin_split(self, labels, feature, data, data_entropy):
        """
        Calculates the best possible binary split out of the possibilities in self.thresholds
        """
        if self.thresholds is None:
            raise RuntimeError('No specified thresholds')

        thresholds = np.array(self.thresholds)
        best_split = None
        best_inf_gain = -math.inf
        best_threshold = None

        # last threshold is math.inf, ignore it when choosing the best split
        for threshold in thresholds[:-1]:
            idx_data_less = np.where(data[:, feature] < threshold)[0]
            idx_data_gte = np.where(data[:, feature] >= threshold)[0]

            split = np.array([idx_data_less, idx_data_gte])
            inf_gain = self.split_inf_gain(data, labels, split, data_entropy)

            if inf_gain > best_inf_gain:
                best_inf_gain = inf_gain
                best_split = split
                best_threshold = threshold

        # last threshold must be math.inf
        best_thresholds = np.array([best_threshold, math.inf])

        return best_thresholds, best_split, best_inf_gain

    def split_inf(self, data, labels, split):
        inf = 0.0
        for subset_ids in split:
            proportion = len(subset_ids) / data.shape[0]
            subset_label_counts = Counter(labels[subset_ids])
            subset_entropy = entropy(subset_label_counts.values())
            inf += proportion * subset_entropy

        return inf

    def split_inf_gain(self, data, labels, split, data_entropy):
        inf = self.split_inf(data, labels, split)
        inf_gain = data_entropy - inf

        return inf_gain

    def id3(self, labels, features, data):

        label_counts = Counter(labels)

        # if there's only one label left in the set, return a leaf with that label
        if len(label_counts) == 1:
            return TreeNode(label=labels[0])

        if len(features) == 0:
            most_common_label = label_counts.most_common(1)[0][0]
            return TreeNode(label=most_common_label)

        data_entropy = entropy(label_counts.values())

        # randomize the order in which feature splits are evaluated
        # since the first pixels hold less information
        rnd_features = list(features)
        random.shuffle(rnd_features)

        # find the best attribute to split on
        best_feature = None
        best_split = None
        best_inf_gain = -math.inf
        best_thresholds = None

        for i, feature in enumerate(rnd_features):

            thresholds, split, inf_gain = self.best_split(labels, feature, data, self.split_method, data_entropy)

            if inf_gain > best_inf_gain:
                best_inf_gain = inf_gain
                best_feature = feature
                best_split = split
                best_thresholds = thresholds

            # consider only up to max_features when searching for the best split
            # except if no valid split has been found, then the search continues
            if i > self.max_features and len(best_thresholds) > 1:
                break

        # if the best split contains only one non-empty bucket, return a leaf
        if len(best_split) == 1:
            most_common_label = label_counts.most_common(1)[0][0]
            return TreeNode(label=most_common_label)

        features.remove(best_feature)
        node = TreeNode(children=list(), feature=best_feature, thresholds=best_thresholds)

        for i, subset_ids in enumerate(best_split):
            data_subset = data[subset_ids]
            labels_subset = labels[subset_ids]

            node.new_child(self.id3(labels_subset, features, data_subset))

        # undo best_feature remove, so the higher branches can still split on that feature
        features.add(best_feature)

        return node

    def predict(self, input, node=None):
        if node is None:
            node = self.root

        if node.label is not None:
            return node.label

        feature_val = input[node.feature]
        for i, threshold in enumerate(node.thresholds):
            if feature_val < threshold:
                return self.predict(input, node.children[i])

        # # if feature_val doesn't satisfy any thresholds it means it's in the last child
        # # e.g if it's a binary split on threshold = 3, a value 5 will be in the second child
        # return self.predict(input, node.children[-1]


class TreeNode:
    def __init__(self, *, children=None, feature=None, thresholds=None, label=None):
        self.children = children
        self.label = label
        self.feature = feature
        self.thresholds = thresholds

    def new_child(self, child):
        self.children.append(child)
