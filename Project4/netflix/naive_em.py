"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    from scipy.stats import multivariate_normal
    n, d = X.shape
    K,_ = mixture.mu.shape  # Number of components in the mixture
    pi = mixture.p  # Weights of the components
    mu = mixture.mu  # Means of the components
    cov = mixture.var  # Covariances of the components

    # Initialize the responsibility matrix (soft counts)
    responsibilities = np.zeros((n, K))

    # Calculate the responsibilities for each data point and each component
    for k in range(K):
        # Use multivariate_normal.pdf to compute the probability density function
        responsibilities[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=cov[k])

    # Normalize the responsibilities to get soft counts
    responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= responsibilities_sum

    # Calculate the log-likelihood of the assignment
    log_likelihood = np.sum(np.log(np.sum(responsibilities_sum * pi, axis=1)))
    
    return responsibilities, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    # Update weights
    weights = post.sum(axis=0) / n
    
    # Update means
    means = np.dot(post.T, X) / post.sum(axis=0)[:, np.newaxis]

    # Update covariances
    covariances = np.zeros((K,))
    
    for k in range(K):
        dist_X=0
        for i in range(n):
            dist_X += (post.T[k,i] * np.linalg.norm(X[i] - means[k,:])**2)
        covariances [k] = np.array(dist_X).sum()
    covariances = covariances / (post.sum(axis=0) * d)
    # Create a new GaussianMixture instance with the updated parameters
    new_mixture = GaussianMixture(means, covariances, weights)

    return new_mixture


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    max_iter = 100
    tol = 1.0e-6
    old_log_likelihood = float('-inf')

    for i in range(max_iter):
        # E-step
        post, log_likelihood = estep(X, mixture)
        # M-step
        new_mixture = mstep(X, post)

        # Check for convergence
        relative_improvement = (log_likelihood - old_log_likelihood) / np.abs(log_likelihood)
        if relative_improvement <= tol:
            break

        # Update for the next iteration
        mixture = new_mixture
        old_log_likelihood = log_likelihood

    return new_mixture, post, log_likelihood