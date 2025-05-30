import numpy as np
from .utils import validation_split, regularized_regression_cost, log_current


class LinearRegression:
    """
    Linear Regression model with L2 regularization.

    This model uses stochastic gradient descent (SGD) to optimize weights
    and intercept with L2 regularization to prevent overfitting.

    Attributes
    ----------
    DEFAULT_EPOCHS : int
        Default number of epochs (complete iterations over the training data).
    DEFAULT_ALPHA : float
        Default learning rate.
    DEFAULT_LAMBDA : float
        Default regularization rate.
    DEFAULT_ERROR_THRESHOLD : float
        Default threshold for validation cost convergence.
    DEFAULT_VALIDATION_SIZE : float
        Default portion of data reserved for validation.
    EXIT : bool
        Indicator to signal early stopping when convergence criteria are met.
    """

    DEFAULT_EPOCHS = 100
    DEFAULT_ALPHA = 0.01
    DEFAULT_LAMBDA = 0.0001
    DEFAULT_ERROR_THRESHOLD = 0.001
    DEFAULT_VALIDATION_SIZE = 0.2

    def __init__(self):
        self.EXIT = False  # Set to True if early stopping criteria are met.

    def convergence_test(self, current_cost, past_cost, error_threshold, k):
        """
        Evaluate convergence based on change in cost over iterations.

        Parameters
        ----------
        current_cost : float
            The cost calculated in the current iteration.
        past_cost : float
            The cost calculated in the previous iteration.
        error_threshold : float
            The minimal expected decrease in cost before considering convergence.
        k : int
            The current epoch number.

        Updates
        -------
        Sets self.EXIT to True if the cost change remains below the threshold
        for 10 consecutive checks.
        """
        if past_cost - current_cost <= error_threshold:
            self.c += 1
            if self.c >= 10:
                self.EXIT = True  # Either vCost has converged or performance degraded.
        else:
            self.c = 0  # Reset counter if cost reduction is sufficient.

    def single_step(self, Xi, yi, m, W, b, alpha, Lambda):
        """
        Perform a single gradient descent update step.

        Computes the prediction error and updates the weights and bias.

        Parameters
        ----------
        Xi : numpy.ndarray
            A single training instance (features vector).
        yi : float
            The true target value for the instance.
        m : int
            Total number of training samples.
        W : numpy.ndarray
            Current weights.
        b : float
            Current bias.
        alpha : float
            Learning rate.
        Lambda : float
            Regularization rate.

        Returns
        -------
        tuple
            Updated weights and bias (W, b).
        """
        y_i = np.dot(Xi, W) + b
        res = yi - y_i

        dJ_dW = np.dot(res, Xi) - Lambda * W
        dJ_db = res.mean()

        W += dJ_dW * alpha / m
        b += dJ_db * alpha

        return W, b

    def fit(
        self,
        X,
        y,
        epochs=DEFAULT_EPOCHS,
        alpha=DEFAULT_ALPHA,
        Lambda=DEFAULT_LAMBDA,
        error_threshold=DEFAULT_ERROR_THRESHOLD,
        validation_size=DEFAULT_VALIDATION_SIZE,
        output_limit=10,
    ):
        """
        Fit the linear regression model to the training data.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix for training.
        y : numpy.ndarray
            Target values.
        epochs : int, optional
            Number of iterations over the training set (default is DEFAULT_EPOCHS).
        alpha : float, optional
            Learning rate for gradient descent (default is DEFAULT_ALPHA).
        Lambda : float, optional
            Regularization rate for L2 regularization (default is DEFAULT_LAMBDA).
        error_threshold : float, optional
            Minimal decrease in cost to avoid early stopping (default is DEFAULT_ERROR_THRESHOLD).
        validation_size : float, optional
            Fraction of the data to reserve for validation (default is DEFAULT_VALIDATION_SIZE).
        output_limit : int, optional
            Frequency control for logging outputs (default is 10).

        Returns
        -------
        tuple
            Optimized weights and bias (W, b).

        Raises
        ------
        ValueError
            If output_limit is less than or equal to 0.
        """
        if output_limit <= 0:
            raise ValueError("Output limit should be greater than 0")

        num_out = epochs // output_limit
        np.set_printoptions(precision=4)

        # Split data into training and validation sets.
        X, y, X_val, y_val = validation_split(X, y, validation_size)
        m, n = X.shape

        # Random initialization of weights and bias.
        W = np.random.rand(n)
        b = np.random.rand()

        # Initial cost evaluation.
        y_ = np.dot(X, W) + b
        y_val_ = np.dot(X_val, W) + b
        cost = regularized_regression_cost(y, y_, Lambda, W, m)
        past_cost = regularized_regression_cost(y_val, y_val_, Lambda, W, m)
        current_cost = 0
        k = 0

        self.c = 0  # Counter for convergence checking.

        log_current(0, num_out, output_limit, cost, past_cost)  # Initial logging.

        try:
            for k in range(1, epochs + 1):
                # Perform a single epoch using stochastic gradient descent (SGD).
                for i in range(m):
                    W, b = self.single_step(X[i], y[i], m, W, b, alpha, Lambda)

                # Log intermediate outputs based on the specified output limit.
                if k % num_out == 0:
                    y_ = np.dot(X, W) + b
                    y_val_ = np.dot(X_val, W) + b
                    cost = regularized_regression_cost(y, y_, Lambda, W, m)
                    vcost = regularized_regression_cost(y_val, y_val_, Lambda, W, m)
                    log_current(k, num_out, output_limit, cost, vcost)

                # Calculate current validation cost for convergence testing.
                y_val_ = np.dot(X_val, W) + b
                current_cost = regularized_regression_cost(y_val, y_val_, Lambda, W, m)

                self.convergence_test(current_cost, past_cost, error_threshold, k)

                if self.EXIT:
                    log_current(
                        k=k,
                        num_out=0,
                        output_limit=0,
                        cost=0,
                        vcost=current_cost,
                        alter=True,
                    )
                    print(
                        f"\nEpoch {k} > Validation cost converged within threshold "
                        f"{error_threshold} or performance degraded."
                    )
                    return W, b

                past_cost = current_cost

        except KeyboardInterrupt:
            log_current(
                k=k, num_out=0, output_limit=0, cost=0, vcost=current_cost, alter=True
            )
            print(f"\nTraining interrupted! Final weights: {W}, Bias: {b}")
            return W, b

        return W, b
