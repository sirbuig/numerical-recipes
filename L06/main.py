# Lab 6 - Bayesian Methods

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

x = [82, 106, 120, 68, 83, 89, 130, 92, 99, 89]
mu, sigma = 90, 10

mean_values = [70, 75, 80, 85, 90, 95, 100]
sigma_values = [5, 10, 15, 20]

def normal_distribution(mu, sigma):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    plt.plot(x, y)
    plt.title(f'Normal Distribution (μ={mu}, σ={sigma})')
    plt.show()


def likelihood(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def apriori(mu, sigma):
    mu_0 = stats.norm.pdf(mu, 100, 50)
    sigma_0 = stats.uniform.pdf(sigma, 1, 70)

    return mu_0 * sigma_0

def best_model():
    results = []
    for mean_ in mean_values:
        for sigma_ in sigma_values:
            results.append((apriori(mean_, sigma_) * np.prod(stats.norm.pdf(x, mean_, sigma_)), mean_, sigma_))

    max_prob = max(results, key=lambda v: v[0])
    return max_prob[0], max_prob[1], max_prob[2]

if __name__ == '__main__':
    print("---------------------\n1)")
    normal_distribution(mu, sigma)

    print("---------------------\n2)")
    print(likelihood(82, mu, sigma), stats.norm.pdf(82, mu, sigma), sep='\n')

    print("---------------------\n3)")
    print(stats.norm.pdf(x, mu, sigma))

    print("---------------------\n4)")
    print(apriori(90, 10))

    print("---------------------\n5)")
    print(apriori(90, 10) * np.prod(stats.norm.pdf(x, mu, sigma)))

    print("---------------------\n6)")
    value, mean, sigma = best_model()
    print(value, mean, sigma)