import cv2 as cv

from .utils import line_transform_img

def load_image_1d(path):
    return line_transform_img(cv.cvtColor(cv.imread('./images/' + str(path) + '.bmp'), cv.COLOR_BGR2GRAY))

alfa2 = load_image_1d('alfa2')

beee2 = load_image_1d('beee2')

cible2 = load_image_1d('cible2')

city2 = load_image_1d('city2')

country2 = load_image_1d('country2')

promenade2 = load_image_1d('promenade2')

veau2 = load_image_1d('veau2')

zebre2 = load_image_1d('zebre2')