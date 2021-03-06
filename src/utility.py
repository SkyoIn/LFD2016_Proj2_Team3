import os
from os import listdir
from os.path import join
import cPickle
import numpy as np
from PIL import Image
from scipy.linalg import norm
from scipy.ndimage.filters import gaussian_filter


def bilinear_interpolate(values, dx, dy):
    """Interpolating with given dx and dy"""
    assert values.shape == dx.shape == dy.shape

    A = np.zeros(values.shape)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            x = i + dx[i, j]
            y = j + dy[i, j]
            if x < 0:
                x = x + int(1 + 0 - x)
            if x >= values.shape[0] - 1:
                x = x - int(1 + x - (values.shape[0] - 1))
            if y < 0:
                y = y + int(1 + 0 - y)
            if y >= values.shape[1] - 1:
                y = y - int(1 + y - (values.shape[1] - 1))

            x1 = int(x)
            y1 = int(y)
            x2 = x1 + 1
            y2 = y1 + 1
            f11 = values[x1, y1]
            f12 = values[x1, y2]
            f21 = values[x2, y1]
            f22 = values[x2, y2]

            A[i, j] = (
                f11*(x2-x)*(y2-y) + f12*(x2-x)*(y-y1)
                + f21*(x-x1)*(y2-y) + f22*(x-x1)*(y-y1)
            )
    return A

def elastic_distortion(image, sigma=5, alpha=36):
    def delta():
        d = gaussian_filter(np.random.uniform(-1, 1, size=image.shape), sigma)
        return (d / norm(d)) * alpha

    assert image.ndim == 2
    dx = delta()
    dy = delta()
    return bilinear_interpolate(image, dx, dy)


def load_images(base_dir, resize_shape=64, mode="train", one_hot=False):
    if mode == "save_and_train":
        with open(os.path.join(os.path.dirname(__file__), '../data', 'train',"X.pkl"), 'rb') as f:
            X = cPickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), '../data', 'train',"Y.pkl"), 'rb') as f:
            Y = cPickle.load(f)

    else:
        image_names = listdir(base_dir)

        X = []
        Y = np.empty(0)
        for i, image_name in enumerate(image_names):
            image_path = join(base_dir, image_name)
            y = image_name.split('-')[0][3:]
            y = int(y)
            image = Image.open(image_path)

            # change to black image
            black_image = image.convert('L')
            # resize the image
            black_resized_image = black_image.resize((resize_shape, resize_shape))
            # black_resized_image.show()
            # image class to array
            im = np.array(black_resized_image, dtype=np.int16, copy=True)

            if mode == "save" or mode == "train":
                 # affine transformation
                im_elastic0 = elastic_distortion(im, sigma=0.5)
                im_elastic1 = elastic_distortion(im, sigma=1)
                im_elastic2 = elastic_distortion(im, sigma=2)
                im_elastic3 = elastic_distortion(im, sigma=3)
                im_elastic4 = elastic_distortion(im, sigma=4)
                im_elastic5 = elastic_distortion(im, sigma=5)
                im_elastic6 = elastic_distortion(im, sigma=6)

                # Image.fromarray(im).convert('RGB').save('original.png', "PNG")
                # Image.fromarray(im_elastic0).convert('RGB').save('sigma0.5.png', "PNG")
                # Image.fromarray(im_elastic1).convert('RGB').save('sigma1.png', "PNG")
                # Image.fromarray(im_elastic2).convert('RGB').save('sigma2.png', "PNG")
                # Image.fromarray(im_elastic3).convert('RGB').save('sigma3.png', "PNG")
                # Image.fromarray(im_elastic4).convert('RGB').save('sigma4.png', "PNG")
                # Image.fromarray(im_elastic5).convert('RGB').save('sigma5.png', "PNG")
                # Image.fromarray(im_elastic6).convert('RGB').save('sigma6.png', "PNG")
                # break
                # Image.fromarray(im_elastic3).show()
            image.close()

            # TODO
            # change below - preprocessing for image file
            # memory will be explode if you are not doing dimension reduction and copy ( 2*1200*900 = 2.06MB, numpy object = 26MB per image)
            shape = im.shape
            im = np.reshape(im, shape[0]*shape[1])
            # im = np.copy(im) # MUST
            # print im.shape
            Y = np.append(Y, y)
            X.append(im)

            if mode == "save" or mode == "train":
                im_elastic0 = np.reshape(im_elastic0, shape[0]*shape[1])
                X.append(im_elastic0)
                Y = np.append(Y, y)

                im_elastic1 = np.reshape(im_elastic1, shape[0]*shape[1])
                X.append(im_elastic1)
                Y = np.append(Y, y)

                im_elastic2 = np.reshape(im_elastic2, shape[0]*shape[1])
                X.append(im_elastic2)
                Y = np.append(Y, y)

                X.append(np.reshape(im_elastic3, shape[0]*shape[1]))
                Y = np.append(Y, y)

                X.append(np.reshape(im_elastic4, shape[0]*shape[1]))
                Y = np.append(Y, y)

                X.append(np.reshape(im_elastic5, shape[0]*shape[1]))
                Y = np.append(Y, y)

                X.append(np.reshape(im_elastic6, shape[0]*shape[1]))
                Y = np.append(Y, y)



        # if mode == "save":
        #     with open(os.path.join(os.path.dirname(__file__), '../data', 'train',"X.pkl"), 'wb+') as f:
        #         cPickle.dump(X, f)
        #     with open(os.path.join(os.path.dirname(__file__), '../data', 'train',"Y.pkl"), 'wb+') as f:
        #         cPickle.dump(Y, f)

    if one_hot is True:
        Y = Y-1
        Y = Y.tolist()
        temp_Y = np.zeros(shape=[len(Y), 62])
        temp_Y[list(range(len(Y))), Y] = 1
        Y = temp_Y

    return np.array(X), Y

if __name__ == "__main__":
    train_dir = os.path.join(os.path.dirname(__file__), '../data', 'train')
    test_dir = os.path.join(os.path.dirname(__file__), '../data', 'test')
    X, Y = load_images(train_dir, mode="save")
