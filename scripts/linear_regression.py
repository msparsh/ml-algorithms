import numpy as np

from utils import validation_split, regularized_regression_cost as compute_cost
from utils import log_current


class LinearRegression:
    """Linear regression model with L2 regularization."""

    DEFAULT_EPOCHS = 100
    DEFAULT_ALPHA = 0.01
    DEFAULT_LAMBDA = 0.0001
    DEFAULT_ERROR_THRESHOLD = 0.001
    DEFAULT_VALIDATION_SIZE = 0.2

    def __init__(self):
        self.EXIT = False

    def convergence_test(self, current_cost, past_cost, error_threshold, k):
        # Simple convergence test
        if past_cost - current_cost <= error_threshold:
            self.c += 1
            if self.c >= 10:
                log_current(k=k, num_out=0, output_limit=0, cost=0, vcost=current_cost, alter=True)
                print(f"\nEpoch {k} > vCost Converged with threshold {error_threshold}. Or performance degraded.")
                self.EXIT = True  # Also returns in case of validation perf degradation (overfit)

        else:
            self.c = 0  # For counting consecutive iterations of convergence

    def single_step(self, Xi, yi, m, W, b, alpha, Lambda):
        """Perform a single step of gradient descent."""

        y_i = np.dot(Xi, W) + b
        res = yi - y_i

        dJ_dW = np.dot(res, Xi) - Lambda * W
        dJ_db = res.mean()

        W += dJ_dW * alpha / m
        b += dJ_db * alpha

        return W, b

    def fit(self, X, y,
            epochs=DEFAULT_EPOCHS,
            alpha=DEFAULT_ALPHA,
            Lambda=DEFAULT_LAMBDA,
            error_threshold=DEFAULT_ERROR_THRESHOLD,
            validation_size=DEFAULT_VALIDATION_SIZE,
            output_limit=10):
        """Fit the linear regression model to the given data.
        
        Parameter
        ---------
        epochs: int, default=1000
            Number of complete iterations through X

        alpha : float, default=0.01
            Constant Learning Rate

        Lambda : float, default=0.0001
            Rate for l2 Regularization
        
        error_threshold: float, default=0.001
            Threshold for vCost convergence
        
        validation_size: float, default=0.2
            Percent of data for validation, 0 <= vs < 1

        output_limit : int, default=10
            Number of iterations to show

        Returns
        -------
        W : numpy.ndarray
            The optimized weights.
        b : numpy.longdouble
            The optimized intercept.
        """

        if output_limit <= 0:
            raise ValueError("Output limit should be greater than 0")

        num_out = epochs // output_limit
        np.set_printoptions(precision=4)

        X, y, X_val, y_val = validation_split(X, y, validation_size)
        m, n = X.shape

        W = np.random.rand(n)
        b = np.random.rand()

        y_ = np.dot(X, W) + b
        y_val_ = np.dot(X_val, W) + b

        cost = compute_cost(y, y_, Lambda, W, m)
        past_cost = compute_cost(y_val, y_val_, Lambda, W, m)
        current_cost = 0
        k=0

        self.c = 0  # to count convergence for consecutive iterations

        log_current(0, num_out, output_limit, cost, past_cost)  # Initial Out

        try:
            for k in range(1, epochs + 1):
                # SGD
                for i in range(m):
                    W, b = self.single_step(X[i], y[i], m, W, b, alpha, Lambda)
                # SGD

                # LOG OUTPUT
                if k % num_out == 0:
                    y_ = np.dot(X, W) + b
                    y_val_ = np.dot(X_val, W) + b

                    cost = compute_cost(y, y_, Lambda, W, m)
                    vcost = compute_cost(y_val, y_val_, Lambda, W, m)

                    log_current(k, num_out, output_limit, cost, vcost)

                # LOG OUTPUT

                # CONVERGENCE
                y_val_ = np.dot(X_val, W) + b
                current_cost = compute_cost(y_val, y_val_, Lambda, W, m)  # vCost

                self.convergence_test(current_cost, past_cost, error_threshold, k)

                if self.EXIT:
                    return W, b

                past_cost = current_cost
                # CONVERGENCE

        # CTRL C
        except KeyboardInterrupt:
            log_current(k=k, num_out=0, output_limit=0, cost=0, vcost=current_cost, alter=True)
            print(f"\nTerminated! Returned: Weights: {W}, Bias: {b}")
            return W, b
        # CTRL C

        return W, b
