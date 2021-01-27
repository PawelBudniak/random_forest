import numpy as np
from collections import Counter
from collections import namedtuple
import math
import statistics
import copy
import test


def entropy(counts):
    total = sum(counts)
    return sum(-count / total * math.log(count / total) for count in counts)


class Tree:
    SPLIT_METHODS = {'mean', 'thresholds'}

    def __init__(self, data, labels, *, features=None, split_method='mean', thresholds=None):
        """
              Mandatory args:
              :param labels: data labels
              :param features: default=None
                     data features, if not specified they are computed as: set(range(data.shape[1]))
              Optional kwargs:
              :param data: training data
              :param split_method: {'mean','thresholds'}, default = 'mean'
                     how to split the data, if 'thresholds' chosen, the thresholds must be set in param thresholds

              """
        if features is None:
            features = set(range(data.shape[1]))  # each column represents one pixel - one feature

        if split_method not in self.SPLIT_METHODS:
            raise ValueError('Possible split methods: ', self.SPLIT_METHODS)
        self.split_method = split_method

        if split_method == 'thresholds':
            if thresholds is None:
                raise RuntimeError('split_method=\'thresholds\' chosen without providing a threshold Iterable')
            self.thresholds = sorted(thresholds)
            # last threshold must be math inf
            if self.thresholds[-1] != math.inf:
                self.thresholds.append(math.inf)

        self.root = self.id3(labels, features, data)

        # self.c45(data, labels, self.root)

    def best_split(self, labels, feature, data, split_method='mean'):
        """

        :param labels: data labels
        :param feature: feature to find a split on
        :param data: training data
        :param split_method: {'mean','thresholds'}, default = 'mean'
               how to split the data, if 'thresholds' chosen, they can be set in param thresholds

        :return: np.array(float): thresholds, last threshold is always math.inf
                ,np.array(np.array): indices of data split on thresholds


        """

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

        # find non-empty subsets
        non_empty_ids = [i for i in range(len(split)) if len(split[i]) > 0]
        split = split[non_empty_ids]
        thresholds = thresholds[non_empty_ids]
        # if there is only one non-empty subset, the threshold to enter it is math.inf
        if len(thresholds) == 1:
            thresholds[0] = math.inf

        return thresholds, split

    def id3(self, labels, features, data):

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
            thresholds, split = self.best_split(labels, feature, data, self.split_method)
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
        if len(best_split) == 1:
            most_common_label = label_counts.most_common(1)[0][0]
            return TreeNode(label=most_common_label)

        features.remove(best_feature)
        node = TreeNode(children=list(), feature=best_feature, thresholds=best_thresholds)

        for i, subset_ids in enumerate(best_split):
            data_subset = data[subset_ids]
            labels_subset = labels[subset_ids]

            node.new_child(self.id3(labels_subset, features, data_subset))

        # undo best_feature remove, so the higher branches can still use it
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

    def c45(self, data, labels, node, filtered_ids=None):
        # if root of the tree or subtree
        if filtered_ids is None:
            filtered_ids = []

        print('wjezdzam')

        # if node is a leaf
        if node.label is not None:
            return node

        if node == self.root:
            test_data = data
            test_labels = labels

        last_thresh = 0
        # error values
        temp_data = np.copy(data)
        temp_labels = np.copy(labels)
        for i, (child, threshold) in enumerate(zip(node.children, node.thresholds)):

            data = np.hstack((data, np.atleast_2d(labels).T))
            # test_data = data[(last_thresh >= data[:, node.feature]) & (data[:, node.feature] < threshold)]
            test_index = np.where(np.logical_and(data[:, node.feature] >= last_thresh, data[:, node.feature] < threshold))
            test_data = data[test_index]
            test_labels = np.transpose(test_data[:, -1])
            test_data = test_data[:, :-1]
            data = data[:, :-1]
            # if not np.array_equal(data, temp_data) or not np.array_equal(labels, temp_labels):
            #     exit(69)

            node.children[i] = self.c45(test_data, test_labels, child, filtered_ids)
            last_thresh = threshold

        print("huj\n")
        # calculate subtree error
        if len(labels) == 0:
            return node
        st_err = test.error_rate(data, labels, node)

        subtree_error = st_err + pow(st_err * (1 - st_err), 0.5) / len(labels)

        # find most common label
        label_counts = Counter(labels)  # .most_common(0)[0][0]
        best_label = label_counts.most_common()[0][0]

        lf_err = 1 - label_counts[best_label] / len(labels)

        leaf_error = lf_err + pow(lf_err * (1 - lf_err), 0.5) / len(labels)

        if subtree_error + 0.05 >= leaf_error:
            print("kurwa gicior taborecior\n")
            return TreeNode(label=best_label)
        print("*hej\n")
        return node


        # leaf_ids = np.where(labels == best_label)[0]


class TreeNode:
    def __init__(self, *, children=None, feature=None, thresholds=None, label=None):
        self.children = children
        self.label = label
        self.feature = feature
        self.thresholds = thresholds

    def new_child(self, child):
        self.children.append(child)

    def predict(self, input, node=None):
        if node is None:
            node = self

        if node.label is not None:
            return node.label

        feature_val = input[node.feature]
        for i, threshold in enumerate(node.thresholds):
            if feature_val < threshold:
                return self.predict(input, node.children[i])
