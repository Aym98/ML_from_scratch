import numpy as np 
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as skclf, DecisionTreeRegressor as skreg
from sklearn.metrics import root_mean_squared_error
import time 

# Decision tree algo : 
#   Select base node (Dataset)
#   Choose best feature  CART vs ID3 
#   Split the node 
#   Are resulting nodes pure ?
#   If no, repeat from step 2 
#   Stop when all nodes are pure, or max depth is reached 

class Criterion:
    def __init__(self, name = "entropy"):
        self.name = name
    
    def compute_proba(self, y):
        return np.bincount(y) / len(y)
    
    def compute_entropy(self, y, epsilon = 1e-20):
        probabilities = self.compute_proba(y)
        log_prob = np.log(probabilities + epsilon)
        return - np.dot(probabilities, log_prob.T)
    
    def compute_gini(self, y):
        probabilities = self.compute_proba(y)
        return 1 - np.dot(probabilities, probabilities.T)
    
    def compute_mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)
    
    def compute_gain(self, y, left, right):
        n = len(y)
        n_l, n_r = len(left), len(right)
        y_left, y_right = y[left], y[right]
        #Check if if the data is not split 
        if n_l * n_r == 0:
            return 0
        # If the criterion is entropy 
        if self.name == "entropy":
            return self.compute_entropy(y) - (n_l/ n) * self.compute_entropy(y_left) - (n_r / n) * self.compute_entropy(y_right)
        # Else if the criterion is gini 
        if self.name == "gini":
            return self.compute_gini(y) - (n_l/ n) * self.compute_gini(y_left) - (n_r / n) * self.compute_gini(y_right)
        # If it's a decision tree regressor
        else:
          return self.compute_mse(y) - (n_l/ n) * self.compute_mse(y_left) - (n_r/ n) * self.compute_mse(y_right)

        
class Node:
    def __init__(self, feature = None, thresh = None, left = None, right = None, value = None):
        self.feature = feature
        self.thresh = thresh
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree():
    def __init__(self, max_depth = 5, min_sample = 2, n_features = None, root = None, min_gain = float(1e-10), criterion = "entropy"):
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.n_features = n_features
        self.min_gain = min_gain
        self.criterion = Criterion(criterion)
        if root is None:
            self.root = Node()

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.grow_tree(X, y)
    
    def get_value(self, y):
        raise NotImplementedError
        
    def grow_tree(self, X, y, depth = 0):
        n_samples, features = X.shape
        nlabels = len(np.unique(y))
        #Checking stopping criteria
        if depth > self.max_depth or n_samples < self.min_sample or nlabels == 1:
            value = self.get_value(y)
            return Node(value = value)
        #Choosing a set of features at random
        feats = np.random.choice(features, size = self.n_features, replace=False)
        #Getting the best split out of those features 
        gain, feature, thresh, left, right = self.get_best_split(X, y, feats)
        if feature is None or gain < self.min_gain:
            return Node(value= self.get_value(y))
        #Growing the left and right side of the tree 
        left = self.grow_tree(X[left, :], y[left], depth + 1)
        right = self.grow_tree(X[right, :], y[right], depth + 1)
        return Node(feature, thresh, left, right)

    def split(self, X_column, thresh):
        l_index = np.where(X_column <= thresh)[0]
        r_index = np.where(X_column > thresh)[0]
        return l_index, r_index  

    def get_best_split(self, X, y, feats):
        best_gain = -1
        best_feat, best_thresh, best_l_index, best_r_index = None, None, None, None
        for feat in feats:
            X_column = X[:, feat]
            threshs = np.unique(X_column)
            for thresh in threshs:
                l_index, r_index = self.split(X_column, thresh)
                info_gain = self.criterion.compute_gain(y, l_index, r_index)
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_thresh = thresh
                    best_feat = feat
                    best_l_index, best_r_index = l_index, r_index
        return best_gain, best_feat, best_thresh, best_l_index, best_r_index 
    
    def predict(self, X):
        values = []
        for x in X:
            values.append(self.traverse_tree(x, self.root))
        return np.array(values)
    
    def traverse_tree(self, x, node):
        # If leaf node return value 
        if node.value is not None:
            return node.value
        feat, thresh = node.feature, node.thresh
        if x[feat] <= thresh:  
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right) 

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth = 5, min_sample = 2, n_features = None, root = None, min_gain = float(1e-10), criterion = "entropy"):
        if criterion not in ["entropy", "gini"]:
            raise ValueError
        super().__init__(max_depth, min_sample, n_features, root, min_gain, criterion)

    def get_value(self, y):
        return Counter(y).most_common(1)[0][0]
    
class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth = 5, min_sample = 2, n_features = None, root = None, min_gain = float(1e-10)):
        super().__init__(max_depth, min_sample, n_features, root, min_gain, criterion = "variance")

    def get_value(self, y):
        return np.mean(y)

if __name__ == "__main__":

    # Classifier benchmarking
 
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

    my_clf = DecisionTreeClassifier(max_depth=10, criterion="entropy")
    my_start = time.time()
    my_clf.fit(X_train, y_train)
    my_prediction = my_clf.predict(X_test)
    my_end = time.time()

    clf = skclf(max_depth=10, criterion="entropy", min_samples_split=2)
    sk_start = time.time()
    clf.fit(X_train, y_train)
    sk_prediction = clf.predict(X_test)
    sk_end = time.time()

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    #Accuracy benchmark 
    my_acc = accuracy(y_test, my_prediction)
    sk_acc = accuracy(y_test, sk_prediction)
    print(f"The accuracy for this classifiers prediction is {my_acc : .2f} vs sklearns classifier {sk_acc : .2f}")
    
    #Time benchmark 
    print(f"The time it takes for this classifiers training and prediction is {my_end - my_start : .2f} seconds vs sklearns classifier {sk_end - sk_start : .2f} seconds")
    
    # Regressor benchmarking 
    data = datasets.load_diabetes()
    X, y = data.data, data.target 
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    my_reg = DecisionTreeRegressor(max_depth=10)
    myreg_start = time.time()
    my_reg.fit(X_train, y_train)
    my_pred = my_reg.predict(X_test)
    myreg_end = time.time()

    reg = skreg(max_depth=10)
    skreg_start = time.time()
    reg.fit(X_train, y_train)
    sk_pred = reg.predict(X_test)
    skreg_end = time.time()

    #MSE benchmarking 
    my_mse = root_mean_squared_error(y_test, my_pred)
    sk_mse = root_mean_squared_error(y_test, sk_pred)

    #Time benchmarking 
    my_time = myreg_end - myreg_start
    sk_time = skreg_end - skreg_start

    print(f"The rmse for the predictions using my regressor on the diabetes dataset is : {my_mse : .2f}, and sklearns regressor predicition is {sk_mse : .2f}")
    print(f"The time it takes for this regressor training and prediction is {my_time : .2f} seconds vs sklearns regressors {sk_time : .2f} seconds")

