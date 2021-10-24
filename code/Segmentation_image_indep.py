from matplotlib import pyplot as plt
import sklearn.cluster
import numpy as np
import scipy
import math

from .utils import MAP_MPM2, bruit_gauss2, calc_probaprio2, taux_erreur, transform_line_in_img
from .estim_param_EM_indep import estim_param_EM_indep
from .images import load_image_1d


def calc_init_kmeans_indep(Y):
    k_means = sklearn.cluster.KMeans(
        n_clusters=2, random_state=0).fit(Y.reshape(-1, 1))

    p1, p2 = calc_probaprio2(k_means.labels_, 0, 1)

    m1, m2 = k_means.cluster_centers_[0][0], k_means.cluster_centers_[1][0]

    sq_dist = np.min(scipy.spatial.distance.cdist(
        Y.reshape(-1, 1), k_means.cluster_centers_), axis=1) ** 2

    var1 = np.sum((sq_dist * (k_means.labels_ == 0))) / \
        np.sum(k_means.labels_ == 0)

    var2 = np.sum((sq_dist * (k_means.labels_ == 1))) / \
        np.sum(k_means.labels_ == 1)

    sig1, sig2 = np.sqrt(var1), np.sqrt(var2)

    return p1, p2, m1, sig1, m2, sig2


def show_images_indep(img_name, m1, sig1, m2, sig2, X_img, Y_img, S_img):
    plt.figure(figsize=(10, 7))
    plt.suptitle(
        f'{img_name} | m1 : {m1}, sig1 : {sig1} ; m2 : {m2}, sig2 : {sig2}')
    plt.subplot(1, 3, 1)
    plt.title('Image originelle')
    plt.axis('off')
    plt.imshow(X_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 2)
    plt.title('Image bruitée')
    plt.axis('off')
    plt.imshow(Y_img, cmap='gray', vmin=np.min(Y_img), vmax=np.max(Y_img))
    plt.subplot(1, 3, 3)
    plt.title('Image segmentée')
    plt.axis('off')
    plt.imshow(S_img, cmap='gray', vmin=0, vmax=255)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def segmentation_image_indep(image_name, m1, sig1, m2, sig2):
    print(
        f'-----------\n Traitement {image_name} | m1 : {m1}, sig1 : {sig1} ; m2 : {m2}, sig2 : {sig2}')

    X = load_image_1d(image_name)
    Y = bruit_gauss2(X, 0, 255, m1, sig1, m2, sig2)

    p1_0, p2_0, m1_0, sig1_0, m2_0, sig2_0 = calc_init_kmeans_indep(Y=Y)
    p1_K, p2_K, m1_K, sig1_K, m2_K, sig2_K = estim_param_EM_indep(
        K=25, Y=Y, p1=p1_0, p2=p2_0, m1=m1_0, sig1=sig1_0, m2=m2_0, sig2=sig2_0)

    S = MAP_MPM2(Y=Y, cl1=0, cl2=255, p1=p1_K, p2=p2_K,
                 m1=m1_K, sig1=sig1_K, m2=m2_K, sig2=sig2_K)
    S_inv = (S == 0) * 255

    X_img = transform_line_in_img(X, int(math.sqrt(len(X))))
    Y_img = transform_line_in_img(Y, int(math.sqrt(len(Y))))

    if taux_erreur(X, S) > taux_erreur(X, S_inv):
        S = S_inv
    S_img = transform_line_in_img(S, int(math.sqrt(len(S))))

    show_images_indep(image_name=image_name, m1=m1, sig1=sig1,
                      m2=m2, sig2=sig2, X_img=X_img, Y_img=Y_img, S_img=S_img)

    return taux_erreur(X, S)
