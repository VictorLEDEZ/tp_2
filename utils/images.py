import numpy as np

def load_images():
    """loads all the images

    Returns:
        [[Number]]: the loaded images
    """
    images = []
    images.append(np.load('./images/alfa2.bmp'))
    images.append(np.load('./images/beee2.bmp'))
    images.append(np.load('./images/cible2.bmp'))
    images.append(np.load('./images/city2.bmp'))
    images.append(np.load('./images/country2.bmp'))
    images.append(np.load('./images/promenade2.bmp'))
    images.append(np.load('./images/veau2.bmp'))
    images.append(np.load('./images/zebre2.bmp'))
    return images
