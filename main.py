import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call

from decision_tree import DecisionTree


# Collecting data on mushrooms
X = []
y = []
file = open("agaricus-lepiota.data.txt", "r")
for line in file:
    if "?" in line:
        continue
    line = line.strip("\n")
    y.append(line[0])
    line = line.split(",")
    X.append(line[1:])


feature_names = ["cap-shape", "cap-surface", "cap-color", "bruises?",
                 "odor", "gill-attachment", "gill-spacing", "gill-size",
                 "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                 "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
                 "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

feature_values = [
    ["b","c","x","f","k","s"],
    ["f","g","y","s"],
    ["n","b","c","g","r","p","u","e","w","y"],
    ["t","f"],
    ["a","l","c","y","f","m","n","p","s"],
    ["a","d","f","n"],
    ["c","w","d"],
    ["b","n"],
    ["k","n","b","h","g","r","o","p","u","e","w","y"],
    ["e","t"],
    ["b","c","u","e","z","r","?"],
    ["f","y","k","s"],
    ["f","y","k","s"],
    ["n","b","c","g","o","p","e","w","y"],
    ["n","b","c","g","o","p","e","w","y"],
    ["p","u"],
    ["n","o","w","y"],
    ["n","o","t"],
    ["c","e","f","l","n","p","s","z"],
    ["k","n","b","h","r","o","u","w","y"],
    ["a","c","n","s","v","y"],
    ["g","l","m","p","u","w","d"]]

labels = ["p", "e"]




# Model validation
my_clf = DecisionTree()
training_data_x, training_data_y, test_data_x, test_data_y = my_clf.train_test_split(X, y)

print("Model validation using cross-validations:")
learn_params = {
    'feature_names': feature_names,
    'impurity_measure': 'entropy',
    'pruning': False,
}
print("Model validation with entropy without pruning")
print(my_clf.cross_val_score(training_data_x, training_data_y, learn_params, cv=3))


learn_params = {
    'feature_names': feature_names,
    'impurity_measure': 'gini',
    'pruning': False,
}
print("Model validation with gini without pruning")
print(my_clf.cross_val_score(training_data_x, training_data_y, learn_params, cv=3))


learn_params = {
    'feature_names': feature_names,
    'impurity_measure': 'entropy',
    'pruning': True,
}
print("Model validation with entropy with pruning")
print(my_clf.cross_val_score(training_data_x, training_data_y, learn_params, cv=3))


learn_params = {
    'feature_names': feature_names,
    'impurity_measure': 'gini',
    'pruning': True,
}
print("Model validation with gini with pruning")
print(my_clf.cross_val_score(training_data_x, training_data_y, learn_params, cv=3))
print("\n Performance measurement:")


# Performance measure for models
learn_params = {
    'feature_names': feature_names,
    'impurity_measure': 'entropy',
    'pruning': False,
}
print("Performance measure with entropy without pruning")
tree = my_clf.learn(training_data_x, training_data_y, **learn_params)
print(my_clf.score(test_data_x, test_data_y, tree))

learn_params = {
    'feature_names': feature_names,
    'impurity_measure': 'gini',
    'pruning': False,
}
print("Performance measure with gini without pruning")
tree = my_clf.learn(training_data_x, training_data_y, **learn_params)
print(my_clf.score(test_data_x, test_data_y, tree))

learn_params = {
    'feature_names': feature_names,
    'impurity_measure': 'entropy',
    'pruning': True,
}
print("Performance measure with entropy with pruning")
tree = my_clf.learn(training_data_x, training_data_y, **learn_params)
print(my_clf.score(test_data_x, test_data_y, tree))


learn_params = {
    'feature_names': feature_names,
    'impurity_measure': 'gini',
    'pruning': True,
}
print("Performance measure with gini with pruning")
tree = my_clf.learn(training_data_x, training_data_y, **learn_params)
print(my_clf.score(test_data_x, test_data_y, tree))


# Problem 5: Comparing with sk-learns decision-tree.

def translate_x(feature_value_list, x):
    """
    replace data-set elements feature-values with corresponding int given a list of features.

    :param feature_value_list: list, list of lists of feature values
    :param x: data-set to tranform
    :return: None
    """
    for x_elem in x:
        for ind,a in enumerate(x_elem):
            place = 0
            for index, elem in enumerate(feature_value_list[ind]):
                if elem == a:
                    place = index
            x_elem[ind] = place


def trans_y(label_list, y):
    """
    replace data-set labels with corresponding int given a list of labels.

    :param label_list: list of possible labels
    :param y: list of labels to tranform
    :return: None
    """
    for index in range(len(y)):
        place = 0
        for label_index in range(len(label_list)):
            if y[index] == label_list[label_index]:
                place = label_index
        y[index] = place


# Transform data to arrays with ints.
training_data_y = np.array(training_data_y)
training_data_x = np.array(training_data_x)
translate_x(feature_values, training_data_x)
trans_y(labels, training_data_y)

test_data_x = np.array(test_data_x)
test_data_y = np.array(test_data_y)
translate_x(feature_values, test_data_x)
trans_y(labels, test_data_y)

# Performance measure for sklearn
clf = DecisionTreeClassifier(criterion="gini")
clf.fit(training_data_x, training_data_y)
print("\nPerformance measure sklearn with gini without pruning")
print(clf.score(test_data_x, test_data_y))


clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(training_data_x, training_data_y)
print("Performance measure sklearn with entropy without pruning")
print(clf.score(test_data_x, test_data_y))

'''
# Generate an img of tree
dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_names,
    class_names=["p", "e"],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
export_graphviz(clf, out_file='tree.dot', feature_names=feature_names, class_names=labels)
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])
'''
