import numpy as np
import kmeans
import common
import naive_em
import em

def kmeans_lowest_cost():
    # Load the 2D toy dataset
    X = np.loadtxt('toy_data.txt')

    # Define the values of K
    K_values = [1, 2, 3, 4]

    for K in K_values:
        best_cost = float('inf')
        best_seed = None
        best_mixture = None
        best_post = None

        # Try different random seeds
        for seed in range(5):
            # Initialize K-means
            kmeans_mixture, kmeans_post, kmeans_cost = kmeans.run(X, *common.init(X, K, seed))

            # Select the solution with the lowest cost
            if kmeans_cost < best_cost:
                best_cost = kmeans_cost
                best_seed = seed
                best_mixture = kmeans_mixture
                best_post = kmeans_post
        title = f'K-means with K={K}, Seed={best_seed}'
        # Save the plot for the best solution
        common.plot(X, best_mixture, best_post, title)

        # Report the lowest cost for this K
        print(f'Lowest cost for K={K}: {best_cost}')

def run_naive_em():
    # Load the 2D toy dataset
    X = np.loadtxt('toy_data.txt')
    K_values = [1, 2, 3, 4]
    best_k = None
    best_bic = float('-inf')
    for K in K_values:        
        best_seed = None
        best_mixture = None
        best_post = None
        max_log_likelihood = float('-inf')
        # Try different random seeds
        for seed in range(5):
            new_mixture, post, log_likelihood = naive_em.run(X, *common.init(X, K, seed))
            if max_log_likelihood < log_likelihood:
                max_log_likelihood = log_likelihood
                best_seed = seed
                best_mixture = new_mixture
                best_post = post
        print(f'max log_likelihood for K={K}: {round(max_log_likelihood,5)}')
        title = f'naive_em with K={K}, Seed={best_seed}'
        # Save the plot for the best solution
        common.plot(X, best_mixture, best_post, title)
        bic_value = common.bic(X, new_mixture, max_log_likelihood)
        if bic_value > best_bic:
            best_k = K
            best_bic = bic_value
    print(f'bic_value for K={best_k}: {round(best_bic,5)}')

if __name__ == "__main__":
    kmeans_lowest_cost()
    run_naive_em()