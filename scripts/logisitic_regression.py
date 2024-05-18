import numpy as np

from sklearn.metrics import f1_score
from matplotlib import pyplot as plt


class LogisticRegression:
    """Logistic Regression implementation.

    This class provides functionalities to perform logistic regression. Check the fit method.
    Expects a shuffled dataset.
    """
    
    e = 1e-35
    
    def __init__(self):
        """use If required to make model persistent."""
        
        pass
    
    def sigmoid_dot(self, X, W, b):
        """Returns sigmoid of (W.X + b)"""
        
        return 1 / (1 + np.exp(-(np.dot(X, W) + b)))
    
    def update(self, X, res, W, b, alpha, m):
        """Updates W, b stochastically for each datapoint."""
        
        #res = y-y_
        for i in range(m):

            dJ_dW = np.dot(res[i],X[i])
            dJ_db = np.mean(res)
            
            W += alpha * dJ_dW / m
            b += alpha * dJ_db / m
        return W, b 
    
    def create_plot(self, costs):
        """Create line plots for cost and validation cost"""
        
        plt.title("Cost v/s vCost")
        plt.plot(np.arange(0,len(costs["cost"])), costs["cost"], label="Cost")
        plt.plot(np.arange(0,len(costs["vcost"])), costs["vcost"], label="vCost")
        plt.legend()
    
    def cost(self, y, y_, e=e):
        """Returns logistic cost between predicted values and true labels."""
        
        m = y.shape[0]
        c = 0
        
        for i in range(m):
            c += y[i] * np.log(y_[i] + e) + (1 - y[i]) * np.log(1 - y_[i] + e)
        return c / (-m)
    
    def fit(self, X, y, iterations=1000, alpha=0.000001, validation_size=0.4):
        """
        Fits the logistic regression model to the training data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (m, n).
        y : np.ndarray
            True labels of shape (m,).
        iterations : int, optional
            Number of iterations for training. Default is 1000.
        alpha : float, optional
            Learning rate. Default is 0.000001
        validation_size : float, optional
            Portion of data to be used for validation from the end of the data

        Returns
        -------
        tuple
            Trained model parameters W and b.
        """
        val = int(X.shape[0]*(1-validation_size))
        X_val, y_val = X[val:], y[val:]
        X, y =  X[:val], y[:val]
        
        costs = {"cost":[], "vcost":[]}
        
        print(X.shape, y.shape, X_val.shape, y_val.shape)
        m, n = X.shape
        W = np.zeros(n)
        b = 0
        
        for k in range(iterations):
            y_ = self.sigmoid_dot(X, W, b)
            
            y_val_ = self.sigmoid_dot(X_val, W, b)
            res = y - y_
            W, b = self.update(X, res, W, b, alpha,m)
            
            costs["cost"].append(self.cost(y, y_))
            costs["vcost"].append(self.cost(y_val,y_val_))
            if k%(iterations//10) == 0:
                print(f"Iteration: {k}",
                      f"Cost: {self.cost(y, y_)}",
                      f"vCost: {self.cost(y_val,y_val_)}",
                      f"f1: {f1_score(y, y_.round()):.5f}",
                      f"vf1: {f1_score(y_val, y_val_.round()):.5f}"
                     )
            
        self.create_plot(costs)
        return W, b, costs
    
    def predict(self, X, W, b):
        """Generates predictions for input data X using trained model parameters W and b.
        
        Returns rounded predictions. Might need to fix.
        """
        
        s = self.sigmoid_dot(X, W, b)
        return s.round()