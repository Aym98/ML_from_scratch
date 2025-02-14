import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error as mse
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class LinearRegressor:
    def __init__(self, max_iter=1000, thresh=1e-5, learning_rate=0.1):
        self.max_iter = max_iter
        self.thresh = thresh
        self.lr = learning_rate
        self.slope = None
        self.bias = 0.0
        self.errors = []
        self.scaler = StandardScaler()

    def fit(self, X:pd.DataFrame, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        n, p = X.shape[0], X.shape[1]
        X = self.scaler.fit_transform(X)
        self.slope  = np.zeros(p)
        previous_loss = float("inf")
        for _ in range(self.max_iter): 

            y_pred = np.dot(X, self.slope) + self.bias

            dw = 2/n * np.dot(X.T, (y_pred - y))
            db = 2/n * np.sum(y_pred - y)
            
            self.slope -= self.lr * dw
            self.bias -= self.lr * db

            current_loss = mse(y_pred, y)
            self.errors.append(current_loss)
            if abs(current_loss - previous_loss) < self.thresh : 
                break
            previous_loss = current_loss
        return self
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = self.scaler.transform(X)
        return np.dot(X, self.slope) + self.bias


if __name__ == "__main__" :
    lr  = LinearRegressor(max_iter=10000000, learning_rate=0.2, thresh=0.0001)
    data = datasets.load_diabetes()
    X, y = data.data, data.target 
    slope, bias = lr.slope, lr.bias
    print(slope, bias)
    
    lr.fit(X, y)
    slope, bias = lr.slope, lr.bias
    print(slope, bias)
    y_hat = lr.predict(X)
    score = mse(y, y_hat)
    #print(f"predicte value {y_hat} and the actual value {y_train}")
    print(f"first 11 errors :{lr.errors[0 : 10]}")
    print(f"last 11 errors :{lr.errors[-12 : -1]}")
    print(f"score: {score}")
    print(r2_score(y, y_hat))

    # Comparison with sklearn module regressor
    
    regressor = LinearRegression().fit(X, y)
    y_pred = regressor.predict(X)

    print(f"Mean squared error: {mse(y, y_pred):.2f}")
    print(f"Coefficient of determination: {r2_score(y, y_pred):.2f}")