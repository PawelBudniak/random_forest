import numpy as np
from collections import Counter
from collections import namedtuple
import math
import statistics
import copy


def entropy(counts):
    total = sum(counts)
    return sum(-count / total * math.log(count / total) for count in counts)


class Tree:

    def __init__(self, data, labels):
        self.data = data
        self.features = set(range(data.shape[1]))  # each column represents one pixel - one feature
        self.labels = labels

        self.root = self.id3(labels, self.features, data)

    def best_split(self, labels, feature, data, choice_method='mean', thresholds=None):
        """

        :param labels: data labels
        :param feature: feature to find a split on
        :param data: training data
        :param choice_method: {'mean','thresholds'}, default = 'mean'
               how to split the data, if 'thresholds' chosen, they can be set in param thresholds
        :param thresholds: Iterable -> int
               if choice_method = 'thresholds', they can be specified in this parameter

        :return: tuple(float): thresholds, last threshold is always math.inf
                ,tuple(np.array): indexes_of_data_split_on_thresholds


        """

        if choice_method == 'mean':
            # threshold = statistics.mean(data[:, feature])
            threshold = 50

            idx_data_less = np.where(data[:, feature] < threshold)[0]
            idx_data_gte = np.where(data[:, feature] >= threshold)[0]

            thresholds = np.array([threshold, math.inf])
            split = np.array([idx_data_less, idx_data_gte])

            # find non-empty subsets
            non_empty = [i for i in range(len(split)) if len(split[i]) > 0]
            split = split[non_empty]
            thresholds = thresholds[non_empty]
            if len(thresholds) == 1:
                thresholds[0] = math.inf

            return thresholds, split

    def id3(self, labels, features, data):

        features = copy.copy(features)
        label_counts = Counter(labels)

        # if there's only one label left in the set, return a leaf with that label
        if len(label_counts) == 1:
            return TreeNode(label=labels[0])

        if len(features) == 0:
            most_common_label = label_counts.most_common(1)[0][0]
            return TreeNode(label=most_common_label)

        data_entropy = entropy(label_counts.values())

        best_feature = None
        best_split = None
        best_inf_gain = 0.0
        best_thresholds = None

        # find the best attribute to split on

        for feature in features:
            # thresholds, split = self.best_split(labels, feature, data, 'mean')
            thresholds, split = self.best_split(labels, feature, data, 'mean')
            inf = 0.0
            for subset_ids in split:
                proportion = len(subset_ids) / data.shape[0]
                subset_label_counts = Counter(labels[subset_ids])
                subset_entropy = entropy(subset_label_counts.values())
                inf += proportion * subset_entropy

            inf_gain = data_entropy - inf

            if inf_gain >= best_inf_gain:
                best_inf_gain = inf_gain
                best_feature = feature
                best_split = split
                best_thresholds = thresholds

        # if the best split contains only one non-empty bucket, return a leaf
        # idk about this
        # if len(best_split) == 1:
        #     most_common_label = label_counts.most_common(1)[0][0]
        #     return TreeNode(label=most_common_label)

        features.remove(best_feature)
        node = TreeNode(feature=best_feature, children=list(), thresholds=best_thresholds)

        for i, subset_ids in enumerate(best_split):
            data_subset = data[subset_ids]
            labels_subset = labels[subset_ids]

            node.new_child(self.id3(labels_subset, features, data_subset))

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
    def __init__(self, children=None, feature=None, thresholds=None, label=None):
        self.children = children
        self.label = label
        self.feature = feature
        self.thresholds = thresholds

    def new_child(self, child):
        self.children.append(child)
