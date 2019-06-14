class Tree(object):
    """
    A class representing a decision-tree.

    A tree object holds information about its parent,
    children and data needed to learn and predict using the ID3 algorithm.

    Attributes:
        feature_split (str): If a tree splits on a feature, this holds that feature.
        choice (str): If this tree-node is a subtree of a feature_split this holds the split-choice.
        label (str): Holds the predicted label for given tree-node.
        children (:[Tree]:): List of subtrees for given tree-node.
        parent (:Tree:): holds the parent of a tree-node.
        error (int): integer to store error-rate for a tree-node when pruning.
    """
    def __init__(self, feature_split=None, children=None, parent=None, choice=None, label=None, error=None):
        self.feature_split = feature_split
        self.choice = choice
        self.label = label
        self.children = children or []
        self.parent = parent
        self.error = error or 0

    def add_child(self, child):
        """Add a child-tree to tree."""
        self.children.append(child)

    def is_root(self):
        """
        Check if is root of a tree.

        :return: True if is root of tree, else False
        """
        return self.parent is None

    def is_leaf(self):
        """
        Check if is leaf of tree.

        :return: True if is leaf, else False
        """
        return not self.children

    def add_error(self):
        """Increments error for tree-node."""
        self.error = self.error + 1

    def __str__(self):
        if self.is_leaf():
            return str("[LEAF: Label "+str(self.label) +" error "+str(self.error)+" by choice " + str(self.choice)+"]")
        childrens = str(len(self.children))
        if self.is_root():
            return 'NODE: Label: {label} Error: {error} Feature to split: {data}, Childrens: {childrens} [{children}]'.format(data=self.feature_split, children=', '.join(map(str, self.children)), childrens=childrens, label=self.label, error=self.error)
        return 'NODE: Label: {label} Error:{error} Choice: {choice}, Feature to split: {data}, Childrens: {childrens} [{children}]'.format(data=self.feature_split, children=', '.join(map(str, self.children)), childrens=childrens, choice=self.choice, label=self.label, error=self.error)
