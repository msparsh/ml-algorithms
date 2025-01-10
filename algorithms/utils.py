import numpy as np


# Linear
def validation_split(X, y, validation_size=0):
    """Splits X and y into train and validation set.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels vector.
        validation_size (float): Proportion of the dataset to include
                                    in the validation split (between 0 and 1).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Training features, training labels, validation features,
             validation labels.
    """
    val = int(X.shape[0] * (1 - validation_size))
    return X[:val], y[:val], X[val:], y[val:]


def regularized_regression_cost(y, y_, Lambda, W, m):
    """Compute cost function with L2 regularization.

    Args:
        y (np.ndarray): True labels.
        y_ (np.ndarray): Predicted labels.
        Lambda (float): Regularization parameter.
        W (np.ndarray): Weights vector.
        m (int): Number of samples.

    Returns:
        float: Regularized cost.
    """
    return np.mean((y - y_) ** 2) + ((np.sum(W ** 2)) * Lambda / (2 * m))


def log_current(k, num_out, output_limit, cost, vcost, alter=False):
    """Log current training information. Alter for exit print.

    Args:
        k (int): Current epoch.
        num_out (int): Number of outputs per epoch.
        output_limit (int): Total number of outputs.
        cost (float): Current cost.
        vcost (float): Current validation cost.
        alter (bool): If True, alters the logging format.
    """
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
    """Returns logistic cost between predicted values and true labels.

    Args:
        true_value (np.ndarray): True labels (0 or 1).
        predictions (np.ndarray): Predicted probabilities.
        e (float): Small value to avoid log(0).

    Returns:
        float: Logistic cost (log-loss).
    """
    true_value = np.array(true_value)
    pred = np.array(predictions)

    # Compute the log-loss using vectorized operations
    c = true_value * np.log(pred + e) + (1 - true_value) * np.log(1 - pred + e)
    c = -np.mean(c)

    return c


# future : simultaneous_shuffle(X,y) and random_sample(X, y, sample_size)
