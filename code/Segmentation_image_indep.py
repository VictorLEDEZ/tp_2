from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import math

from .utils import MAP_MPM2, bruit_gauss2, calc_probaprio2, taux_erreur, transform_line_in_img
from .classes_calculations import classes_calculations
from .estim_param_EM_indep import estim_param_EM_indep
from .images import load_image_1d

from variables import ITERATIONS, MU1_INITIAL, MU2_INITIAL, SIGMA1_INITIAL, SIGMA2_INITIAL


def segmentation_image_indep(image_name, m1, sig1, m2, sig2):
    print(
        f'-----------\n Traitement {image_name} | m1 : {m1}, sig1 : {sig1} ; m2 : {m2}, sig2 : {sig2}')

    # ! Charge et bruite le signal d'entrée
    X = load_image_1d(image_name)

    class1, class2 = classes_calculations(X)

    Y = bruit_gauss2(X, class1, class2, m1, sig1, m2, sig2)

    # ! Calcule les parametres initiaux
    kmeans = KMeans(n_clusters=2, random_state=0).fit(Y.reshape(-1, 1))

    class1_K, class2_K = classes_calculations(kmeans.labels_)

    p1_initial, p2_initial = calc_probaprio2(
        kmeans.labels_, class1_K, class2_K)

    m1_initial = Y[np.where(kmeans.labels_ == class1_K)].mean()
    m2_initial = Y[np.where(kmeans.labels_ == class2_K)].mean()

    sig1_initial = Y[np.where(kmeans.labels_ == class1_K)].std()
    sig2_initial = Y[np.where(kmeans.labels_ == class2_K)].std()

    # ! Estime les nouveaux parametres
    p1_K, p2_K, m1_K, sig1_K, m2_K, sig2_K = estim_param_EM_indep(
        ITERATIONS, Y, p1_initial, p2_initial, m1_initial, sig1_initial, m2_initial, sig2_initial)

    S = MAP_MPM2(Y, class1, class2, p1_K, p2_K, m1_K, sig1_K, m2_K, sig2_K)

    print(X)

    S_inverse = (S == class1) * class2

    X_image = transform_line_in_img(X, int(math.sqrt(len(X))))
    Y_image = transform_line_in_img(Y, int(math.sqrt(len(Y))))

    if taux_erreur(X, S) > taux_erreur(X, S_inverse):
        S = S_inverse
    S_image = transform_line_in_img(S, int(math.sqrt(len(S))))
    print(S)

    # ! Affiche les images
    plt.figure(figsize=(10, 7))

    plt.subplot(1, 3, 1)
    plt.title('Image originelle')
    plt.axis('off')
    plt.imshow(X_image, cmap='gray', vmin=class1, vmax=class2)

    plt.subplot(1, 3, 2)
    plt.title('Image bruitée')
    plt.axis('off')
    plt.imshow(Y_image, cmap='gray', vmin=np.min(
        Y_image), vmax=np.max(Y_image))

    plt.subplot(1, 3, 3)
    plt.title('Image segmentée')
    plt.axis('off')
    plt.imshow(S_image, cmap='gray', vmin=class1, vmax=class2)

    plt.show(block=True)
    plt.pause(3)
    plt.close()

    return taux_erreur(X, S)


segmentation_image_indep('alfa2', MU1_INITIAL,
                         SIGMA1_INITIAL, MU2_INITIAL, SIGMA2_INITIAL)
