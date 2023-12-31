import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    n, d = X.shape
    m, _ = Y.shape
    kernel_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            dot_product = np.dot(X[i], Y[j])
            kernel_value = (dot_product + c) ** p
            kernel_matrix[i, j] = kernel_value

    return kernel_matrix



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    n, d = X.shape
    m, _ = Y.shape
    kernel_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            # Calculate the squared Euclidean distance between X[i] and Y[j]
            squared_distance = np.sum((X[i] - Y[j]) ** 2)
            # Compute the kernel value using the Gaussian RBF formula
            kernel_value = np.exp(-gamma * squared_distance)
            kernel_matrix[i, j] = kernel_value

    return kernel_matrix
