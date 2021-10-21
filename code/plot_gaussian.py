from matplotlib import pyplot as plt
import scipy.stats as stats
import numpy as np

from .title import title

def calculate_x(mu, sigma):
    return np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

def plot_gaussian(mu1, sigma1, mu2, sigma2):
    """plots the results thanks to pyplot

    Args:
        errors ([[Number]]): all the mean errors depending on the mus and sigmas
        mu1 ([Number]): the array containing the first mus
        sigma1 ([Number]): the array containing the first sigmas
        mu2 ([Number]): the array containing the second mus
        sigma2 ([Number]): the array containing the second sigmas
    """
    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Gaussian distributions')

    x = calculate_x(mu1, sigma1)
    axs[0, 0].plot(x, stats.norm.pdf(x, mu1, sigma1))
    x = calculate_x(mu2, sigma2)
    axs[0, 0].plot(x, stats.norm.pdf(x, mu2, sigma2), 'r')
    axs[0, 0].set_title(title(mu1, sigma1, mu2, sigma2))

    x = calculate_x(mu1, sigma1)
    axs[0, 1].plot(x, stats.norm.pdf(x, mu1, sigma1))
    x = calculate_x(mu2, sigma2)
    axs[0, 1].plot(x, stats.norm.pdf(x, mu2, sigma2), 'r')
    axs[0, 1].set_title(title(mu1, sigma1, mu2, sigma2))

    x = calculate_x(mu1, sigma1)
    axs[1, 0].plot(x, stats.norm.pdf(x, mu1, sigma1))
    x = calculate_x(mu2, sigma2)
    axs[1, 0].plot(x, stats.norm.pdf(x, mu2, sigma2), 'r')
    axs[1, 0].set_title(title(mu1, sigma1, mu2, sigma2))

    x = calculate_x(mu1, sigma1)
    axs[1, 1].plot(x, stats.norm.pdf(x, mu1, sigma1))
    x = calculate_x(mu2, sigma2)
    axs[1, 1].plot(x, stats.norm.pdf(x, mu2, sigma2), 'r')
    axs[1, 1].set_title(title(mu1, sigma1, mu2, sigma2))

    x = calculate_x(mu1, sigma1)
    axs[2, 0].plot(x, stats.norm.pdf(x, mu1, sigma1))
    x = calculate_x(mu2, sigma2)
    axs[2, 0].plot(x, stats.norm.pdf(x, mu2, sigma2), 'r')
    axs[2, 0].set_title(title(mu1, sigma1, mu2, sigma2))

    plt.show()