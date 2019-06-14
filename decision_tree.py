import numpy as np

from tree import Tree


class DecisionTree(object):
    def entropy(self, y_col):
        """
        Calculates the entropy for elements of y_col.

        Entropy is calculated by the Shannon entropy function.

        :param y_col: A list of items
        :return: float, entropy for given column
        """
        y_names_list, y_count_list = np.unique(y_col, return_counts=True)
        ntotal = np.sum(y_count_list)
        entropy = 0
        for y_count in y_count_list:
            entropy += (y_count / ntotal) * np.log2(y_count / ntotal)
        return -entropy

    def gini_index(self, y_col):
        """
        Calculates the gini-index for a y_col.

        :param y_col: A list of items
        :return: float, entropy for given column
        """
        y_names_list, y_count_list = np.unique(y_col, return_counts=True)
        ntotal = np.sum(y_count_list)
        gini = 0
        for y_count in y_count_list:
            gini += pow(y_count / ntotal, 2)
        return 1 - gini

    def information_gain(self, x, y, feature, calc_func):
        """
        Calculates the information-gain going from the data-set x to a subset with given feature,

        Impurity calculated by the function calc_func. e.g. gini or entropy. features can be a list

        :param x: list, dataset
        :param y: list, dataset-labels
        :param feature: int, feature to split on
        :param calc_func: func, method of measurement
        :return: float, the information-gain given dataset and split feature
        """
        parent_impurity = calc_func(y)
        feature_values, values_count = np.unique(x[:, feature], return_counts=True)
        children_impurity = 0
        n_total = np.sum(values_count)
        for i in range(len(values_count)):
            p_childs = values_count[i] / n_total
            target_col = [y[index] for index, x_elem in enumerate(x) if x_elem[feature] == feature_values[i]]
            val_child = calc_func(target_col)
            children_impurity += p_childs * val_child
        return parent_impurity - children_impurity

    def create_tree(self, current_x, current_y, current_feature_list, impurity_measure, parent_node=None):
        """
        A recursive method for generating decision-tree, using the ID3 algorithm.

        :param current_x: list, available dataset
        :param current_y: list, aavailable dataset labels
        :param current_feature_list: list, available features
        :param impurity_measure: string, impurity measure to generate tree
        :param parent_node: Tree, parent tree for this iteration.
        :return: Tree, A decision tree
        """
        if np.unique(current_y).size == 1:  # If pure subset, add leaf
            return Tree(label=np.unique(current_y)[0], parent=parent_node)

        if current_feature_list.size == 0 or len(
                np.unique(current_x)) == 1:  # If impure subset and not able to split any further, or all x_elem equal
            label_list, label_count = np.unique(current_y, return_counts=True)
            max_label = label_list[np.argmax(label_count)]
            return Tree(label=max_label, parent=parent_node)

        if impurity_measure == "entropy":
            calc_function = self.entropy
        elif impurity_measure == "gini":
            calc_function = self.gini_index
        else:
            raise Exception("Unknown impurity measure")

        feature_gains = [self.information_gain(current_x, current_y, feature, calc_function) for feature in
                         current_feature_list]

        max_index = feature_gains.index(max(feature_gains))
        best_feature = current_feature_list[max_index]

        if current_feature_list.size == 1:
            current_feature_list = np.array([])
        else:
            current_feature_list = np.delete(current_feature_list, max_index)

        parent = Tree(feature_split=best_feature, parent=parent_node)

        child_labels = []
        for branch_option in np.unique(current_x[:, best_feature]):
            child_x = [entry for entry in current_x if entry[best_feature] == branch_option]
            child_y = [current_y[ind] for ind, entry in enumerate(current_x) if entry[best_feature] == branch_option]
            child_x = np.array(child_x)
            child_y = np.array(child_y)
            if child_y.size > 0:
                child = self.create_tree(child_x, child_y, current_feature_list, impurity_measure, parent)
                child.choice = branch_option
                parent.add_child(child)
                child_labels.append(child.label)

        labels, counts = np.unique(child_labels, return_counts=True)
        most_common_label = labels[np.argmax(counts)]
        parent.label = most_common_label

        return parent

    def learn(self, x, y, feature_names, impurity_measure='entropy', pruning=False, shuffle=True, prune_size=0.4):
        """
        Generates a decision-tree based on dataset X with labels y.

        Uses ID3 algorithm, and can use gini and entropy as impurity measure.
        Pruning based on reduced-error pruning available. Will then split data in learn and pruning data.

        :param x: list, dataset
        :param y: list, labels for dataset
        :param feature_names: list, features for dataset
        :param impurity_measure: string, type of impurity measure used in ID3
        :param pruning: bool, if pruning or not
        :param shuffle: bool, if shuffle dataset or not
        :param prune_size: float, How much of data used for pruning. Must be float between 0 - 1.
        :return: Tree, a decision-tree
        """

        assert 0 < prune_size <= 1

        feature_names = np.array([i for i in range(len(feature_names))])  # Map features to ints
        x = np.array(x)
        y = np.array(y)

        if pruning:
            if shuffle:
                x_and_y = list(zip(x, y))
                np.random.shuffle(x_and_y)
                x[:], y[:] = zip(*x_and_y)

            # Split data into two subsets
            prune_index = int(len(x) * prune_size)
            x_train = np.array(x[prune_index:])
            x_prune = np.array(x[:prune_index])
            y_train = np.array(y[prune_index:])
            y_prune = np.array(y[:prune_index])

            tree = self.create_tree(x_train, y_train, feature_names, impurity_measure)

            for i in range(len(x_prune)):  # add error for all pruning-examples
                self.insert_error(x_prune[i], y_prune[i], tree)
            self.prune(tree)
        else:
            tree = self.create_tree(x, y, feature_names, impurity_measure)
        return tree

    def predict(self, x, tree):
        """
        Predicts the label for x given a decision-tree.

        :param x: list, a element to predict
        :param tree: Tree, a decision-tree to use for prediction
        :return: string, the predicted label for x
        """
        if tree.is_leaf():
            return tree.label
        for child in tree.children:
            if x[tree.feature_split] == child.choice:
                return self.predict(x, child)
        return tree.label

    def prune(self, tree):
        """
        Prunes a decision-tree based on the reduced-error pruning algorithm.

        Assumes that all tree-nodes have a label, and that their error is updated based on a pruning-dataset.

        :param tree: Tree, The decision-tree to be pruned
        """
        node_error = tree.error
        if tree.is_leaf():
            return node_error
        child_error = 0
        can_delete = True
        for child in tree.children:
            error = self.prune(child)
            if not child.children:
                can_delete = False
            child_error += error
        if node_error <= child_error and can_delete:
            tree.children = []
        return node_error

    def insert_error(self, x, y, tree):
        """
        Increments error in decision-tree for x given label y.

        Recursive. Will start at root of tree and check every tree-node label on its path down to its leaf.
        Increments the error variable for every node.label != y.

        :param x: list, element to insert error for
        :param y: string, label for element
        :param tree: Tree, tree to insert error to
        """
        if tree.label != y:
            tree.add_error()

        if tree.is_leaf():
            return

        for child in tree.children:
            if x[tree.feature_split] == child.choice:
                self.insert_error(x, y, child)

    def cross_val_score(self, x, y, learn_params, cv=3):
        """
        Get an accuracy score for a data-set given learn_params using cross-validation.

        Splits a data-set into cv chuncks. Use one as validation-data and rest as training-data. iterates over and
        returns the mean score of predictions on validation-data for all iterations.

        :param x: list, list of elements
        :param y: list, list of labels for elements
        :param learn_params: dict, dictionary of paramters for learn-function
        :param cv: int, how many chucks/iterations to do cross-val over.
        :return: float, accuracy for given cross-validation
        """
        # Shuffle data
        x_and_y = list(zip(x, y))
        np.random.shuffle(x_and_y)

        list_of_chuncks = np.array_split(x_and_y, cv)
        precision = 0.0
        for i in range(len(list_of_chuncks)):
            validation_chunck_x, validation_chunck_y = zip(*list_of_chuncks[i])
            training_chunck_x, training_chunck_y = zip(
                *np.concatenate([el for index, el in enumerate(list_of_chuncks) if index != i]))
            root = self.learn(training_chunck_x, training_chunck_y, **learn_params)  # TODO: implement dict
            score = 0
            for i in range(len(validation_chunck_x)):
                prediction = self.predict(validation_chunck_x[i], root)
                if prediction == validation_chunck_y[i]:
                    score += 1
            score /= len(validation_chunck_x)
            precision += score
        precision /= len(list_of_chuncks)
        return precision

    def train_test_split(self, x, y, test_size=0.25, shuffle=True):
        """
        Split data into training and test subsets.

        :param x: list, set of data
        :param y: list, labels for data
        :param test_size: float, percentage to use as test-data, must be between 0-1
        :param shuffle: bool, if shuffle or not
        :return: list, list of sets
        """
        x_and_y = list(zip(x, y))
        if shuffle:
            np.random.shuffle(x_and_y)
        split = int(len(x_and_y) * test_size)
        training_set_x, training_set_y = zip(*x_and_y[split:])
        test_set_x, test_set_y = zip(*x_and_y[:split])
        return [training_set_x, training_set_y, test_set_x, test_set_y]

    def score(self, x, y, tree):
        """
        Gives the accuracy for a given dataset and tree.

        :param x: list, list of data
        :param y: list, list of labels for data
        :param tree: Tree, tree to use for prediction
        :return: float, accuracy for predictions.
        """
        score = 0
        for i in range(len(x)):
            if self.predict(x[i], tree) == y[i]:
                score += 1
        return score / len(x)
