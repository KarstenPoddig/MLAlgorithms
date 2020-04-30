import numpy as np

# create some data
x = np.random.randn(20000).reshape(-1, 2) + 5
z = 3*x[:, 0] + (-2)*x[:, 1]
quantile = np.percentile(a=z, q=36)
y = (z > quantile).astype('int')

from sklearn import tree
from sklearn.metrics import precision_score, recall_score, f1_score

clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(x, y)
y_pred = clf.predict(x)


print('Precision: ' + str(precision_score(y, y_pred)))
print('Recall: ' + str(recall_score(y, y_pred)))
print('F1-Score: ' + str(f1_score(y, y_pred)))
print()


# Using those arrays, we can parse the tree structure:

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print()
