import numpy as np


# Linear


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

# Logistic

def logistic_cost(true_value, predictions, e=1e-35):
    """Returns logistic cost between predicted values and true labels."""
    #
    # m = true_value.shape[0]
    # c = 0
    # for i in range(m):
    #     c += true_value[i] * np.log(predictions[i] + e) + (1 - true_value[i]) * np.log(1 - predictions[i] + e)
    # return c / (-m)

    true_value = np.array(true_value)
    predictions = np.array(predictions)

    # Compute the log-loss using vectorized operations
    c = true_value * np.log(predictions + e) + (1 - true_value) * np.log(1 - predictions + e)
    c =-np.mean(c)

    return c


# Stuff to use in future

def simultaneous_shuffle(X, y):
    pass


def random_sample(X, y, sample_size):
    pass
