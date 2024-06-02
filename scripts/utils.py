import numpy as np

# LR


def validation_split(X, y, validation_size=0):
    """Splits X and y into train and validation set"""
    val = int(X.shape[0] * (1 - validation_size))
    return X[:val], y[:val], X[val:], y[val:]


def regularized_regression_cost(y, y_, Lambda, W, m):
    """Compute cost function with L2 regularization."""
    return np.mean((y - y_) ** 2) + ((np.sum(W ** 2)) * Lambda / (2 * m))


def log_current(k, num_out, output_limit, cost, vcost, alter=False):
    """Log current training information. Alter for exit print."""
    if alter:  # For printing at arbitrary epoch, w vCost only
        print(f"       * Epoch: {k}",
              f"vCost: {vcost:.8f}")
        return None

    print(f"({k // num_out}/{output_limit}) > Epoch: {k}",
          f"cost: {cost:.8f}",
          f"vCost: {vcost:.8f}")
# LR
