# Design Decision Record (DDR)

## DDR-001: Stochastic Gradient Descent for Linear Regression

### Context
We are implementing Stochastic Gradient Descent (SGD) for training a Linear Regression model. The goal is to optimize the weights and bias parameters of the linear regression model to minimize the Mean Squared Error (MSE) between the predicted values and the true labels.

### Decision
We have decided to implement the Stochastic Gradient Descent (SGD) optimization algorithm for training the linear regression model. SGD is chosen due to its efficiency and effectiveness in handling large datasets and iterative training. Regularization is skipped for reducing complexity.

### Rationale
- **Efficiency:** SGD updates the model parameters using a single data point (or a small batch) at a time, which makes it computationally efficient, especially for large datasets.
   
- **Memory Requirements:** Unlike batch gradient descent, which requires storing the entire dataset in memory, SGD only requires storing a single data point at a time, reducing memory requirements.

- **Convergence:** SGD can converge faster compared to batch gradient descent since it updates the parameters more frequently, potentially finding the optimal solution quicker.

- **Regularization:** SGD can naturally incorporate regularization techniques such as L2 regularization by adding a regularization term to the gradient update.

### Consequences
- **Stochasticity:** The updates in SGD are noisy due to the random selection of data points, which can lead to fluctuating loss during training.

- **Hyperparameter Tuning:** SGD introduces additional hyperparameters such as the learning rate (alpha) and the number of iterations, which need to be tuned carefully for optimal performance.

- **Convergence:** While SGD can converge faster, it might oscillate around the optimal solution due to its stochastic nature, requiring careful monitoring of the learning process.

### Status
The decision to implement SGD for training the linear regression model has been made based on its efficiency, scalability, and ability to handle large datasets. By implementing SGD, we aim to optimize the model parameters effectively while minimizing computational and memory overheads.