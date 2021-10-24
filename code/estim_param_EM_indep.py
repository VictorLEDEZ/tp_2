import scipy.stats as stats
import numpy as np


def estim_param_EM_indep(iterations, Y, p1_initial, m1_initial, sig1_initial, p2_initial, m2_initial, sig2_initial):

    for i in range(iterations):

        a = p1_initial * stats.norm.pdf(Y, m1_initial, sig1_initial)
        b = p2_initial * stats.norm.pdf(Y, m2_initial, sig2_initial)

        p1_new = a / (a+b)
        p2_new = b / (a+b)

        p1 = np.mean(p1_new)
        p2 = 1 - p1

        m1 = np.sum(Y*p1_new) / np.sum(p1_new)
        m2 = np.sum(Y*p2_new) / np.sum(p2_new)

        sig1 = np.sqrt(np.sum(((Y-m1)**2)*p1_new) / np.sum(p1_new))
        sig2 = np.sqrt(np.sum(((Y-m2)**2)*p2_new) / np.sum(p2_new))

    return p1, m1, sig1, p2, m2, sig2
