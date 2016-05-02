from os import listdir
from os.path import join

import numpy as np
from PIL import Image


def load_images(base_dir):
    image_names = listdir(base_dir)

    X = []
    Y = np.empty(0)
    for i, image_name in enumerate(image_names):
        image_path = join(base_dir, image_name)
        y = image_name.split('-')[0][3:]
        y = int(y)

        image = Image.open(image_path)
        im = np.array(image, dtype=np.int16, copy=True).transpose((1,0,2))
        image.close()

        # TODO
        # change below - preprocessing for image file
        # memory will be explode if you are not doing dimension reduction and copy ( 2*1200*900 = 2.06MB, numpy object = 26MB per image)
        shape = im.shape
        im = np.reshape(im, shape[0]*shape[1]*shape[2])
        im = np.copy(im[0:10]) # MUST
        Y = np.append(Y, y)
        X.append(im)

    return np.array(X), Y



