# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Wed Aug 28 08:28:51 2019
"""


import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import skimage.feature as skf
from scipy import ndimage as ndi


class Image_processor():
    """
    A class to wrap all image filtering and preprocessing. All functions
    are wrapped around skimage. Use path to load an image from file or
    img to load from an array.


    Image_processor(path='', img=np.array([])

    Images are stored in an internal dictionarry 'imgs'. The 'curr' entry is
    where all filters are applied and the key passed to every method is where
    to store the resultant image. This way many filters can be stacked together
    in 'curr' until the final image is stored seperatly under a provided key.
    """

    def __init__(self, path='', img=np.array([])):

        # load the image if path given
        if path:
            img = ski.io.imread(path)[:, :, :3] # exclude any alpha channels
        else:
            assert len(img) > 0, 'Must either pass a path or image'

        # validate the image
        assert isinstance(img, np.ndarray), 'img must be an array'
        assert img.ndim in [2, 3], 'image must be scalar or 3 channel'

        # store the image
        float_img = ski.util.img_as_float(img)
        self.imgs = {'original':float_img, 'curr':float_img.copy()}
        self._dims = np.array(img.shape[:2])


    # ==========================================
    # general purpose functions
    # ==========================================

    def save(self, path, key='curr'):
        "save the given key on the given path"
        ski.io.imsave(path, self.imgs[key])


    def plot(self, key='curr', ax=None):
        "Plot an image"
        if not ax:
            size = 10 * self._dims[::-1] / self._dims[0]
            _fig, ax = plt.subplots(figsize=size)

        col = ax.imshow(self.imgs[key])

        if self.imgs[key].ndim == 2:
            plt.colorbar(col, ax=ax, shrink=.8)

    def reset(self, key='original'):
        "Reset the working image with the given key"
        self.imgs['curr'] = self.imgs[key]


    def rebase(self):
        "mask the working image the new base"
        self.imgs['original'] = self.imgs['curr'].copy()


    def store(self, array, key):
        "add a given array to the imgs"
        self.imgs[key] = array


    # ==========================================
    # morphing functions
    # ==========================================

    def normalise(self, key='curr', std=1.):
        "Take single channel image and normalise it to have mean=0, given std"
        # validate curr image
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        self.imgs[key] = (std * (self.imgs['curr'] - self.imgs['curr'].mean())
                          / self.imgs['curr'].std())
        return self.imgs[key]


    def dilation(self, key='curr', size=3):
        "Apply dilation on a grey scale image"
        # validate curr image
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        selem = np.ones([size, size])
        self.imgs[key] = ski.morphology.dilation(self.imgs['curr'], selem)
        return self.imgs[key]


    def erosion(self, key='curr', size=3):
        "Apply errosion on a grey scale image"
        # validate curr image
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        selem = np.ones([size, size])
        self.imgs[key] = ski.morphology.erosion(self.imgs['curr'], selem)
        return self.imgs[key]


    def median(self, key='curr', **kwargs):
        """
        Apply a median filter on a single channel image with addtitonal
        kwargs passed to skimage.filters.median
        """
        # validate curr image
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        int_img = (self.imgs['curr'] * 255).astype(np.uint8)
        self.imgs[key] = ski.filters.median(int_img, **kwargs)
        return self.imgs[key]


    # ==========================================
    # colorspace functions
    # ==========================================


    def grey_scale(self, key='curr'):
        "Convert an RGB image to grey scale (single channel)"
        self.imgs[key] = ski.color.rgb2gray(self.imgs['curr'])
        return self.imgs[key]


    def hsv(self, key='curr'):
        "Convert an RGB image to hsv scale (both 2 channel)"
        self.imgs[key] = ski.color.rgb2hsv(self.imgs['curr'])
        return self.imgs[key]


    def select_channel(self, channel, key='curr'):
        "Take a single channel of a multi-channel image"
        assert self.imgs['curr'].ndim == 3, 'curr must be a multi-channel image'
        self.imgs[key] = self.imgs['curr'][:, :, channel]
        return self.imgs[key]


    # ==========================================
    # image filters functions
    # ==========================================


    def threshold(self, value, key='curr'):
        "threshold a single channel image to give a binary image"
        # validate curr image
        assert self.imgs['curr'].ndim == 2, 'curr must be a single-channel'

        self.imgs[key] = (self.imgs['curr'] > value).astype(float)
        return self.imgs[key]


    def gabor_filters(self, frequency, n_angs, key='curr'):
        """
        Apply gabor filters of the given frequency and n_angles unfiormly
        distibuted between [0, 180] deg. All the angles are average to
        give a single channel output.
        """
        # validate curr image
        assert self.imgs['curr'].ndim == 2, 'curr must be a single-channel'

        # setup
        thetas = np.linspace(0, np.pi, n_angs)
        gabor_imgs = []

        # calculate
        for ang in thetas:
            kernel = np.real(ski.filters.gabor_kernel(frequency, theta=ang))
            gabor_imgs.append(ndi.convolve(self.imgs['curr'],
                                           kernel, mode='reflect'))
        self.imgs[key] = np.dstack(gabor_imgs).mean(axis=-1)
        return self.imgs[key]


    def gauss(self, sigma=1, key='curr'):
        """
        Apply guassian blur with sigma standard deviation on a multi-channel
        image
        """
        # validate current image
        assert self.imgs['curr'].ndim == 3, 'curr image must be multi-channel'
        self.imgs[key] = ski.filters.gaussian(self.imgs['curr'], sigma=sigma,
                                              multichannel=True)
        return self.imgs[key]


    def gauss_grey(self, sigma=1, key='curr'):
        """
        Apply guassian blur with sigma standard deviation on a single-channel
        image (i.e. grey scale image)
        """
        # validate current image
        assert self.imgs['curr'].ndim == 2, 'curr image must be grey_scale'
        self.imgs[key] = ski.filters.gaussian(self.imgs['curr'], sigma=sigma,
                                              multichannel=False)
        return self.imgs[key]


    def sobel(self, key='curr'):
        """
        Apply sobel edge detection filters on a multi-channel image.
        """
        # validate current image
        assert self.imgs['curr'].ndim == 3, 'curr image must be multi-channel'
        sobs = [ski.filters.sobel(self.imgs['curr'][:, :, i])
                for i in range(3)]
        self.imgs[key] = np.dstack(sobs)
        return self.imgs[key]


    def scharr(self, key='curr'):
        """
        Apply scharr edge detection filters on a multi-channel image.
        """
        # validate current image
        assert self.imgs['curr'].ndim == 3, 'curr image must be multi-channel'
        schs = [ski.filters.scharr(self.imgs['curr'][:, :, i])
                for i in range(3)]
        self.imgs[key] = np.dstack(schs)
        return self.imgs[key]


    def laplace(self, size=3, key='curr'):
        """
        Apply laplace gradient filters of the given size on single or
        multi-chanel images
        """
        int_img = (self.imgs['curr'] * 255).astype(np.uint8)
        self.imgs[key] = ski.filters.laplace(int_img, ksize=size)
        return self.imgs[key]


    # ==========================================
    # complext skimage functions
    # ==========================================


    def lbp(self, radius=3, method='uniform', key='curr'):
        """
        Apply local binary pattern over a square region of the given radius
        to a grey scale image
        """
        # validate curr image
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        self.imgs[key] = skf.local_binary_pattern(self.imgs['curr'],
                                                  radius*8,
                                                  radius,
                                                  method)
        return self.imgs[key]


    def hog(self, key='curr'):
        "Apply histogram of gradients to either single or multi-channel images"
        self.imgs[key] = ski.feature.hog(self.imgs['curr'], visualize=True,
                                         block_norm='L1')[1]
        return self.imgs[key]


    def canny(self, key='curr'):
        "Apply canny edge detection routine on a single channel image"
        # validate curr image
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        self.imgs[key] = ski.feature.canny(self.imgs['curr']).astype(float)
        return self.imgs[key]


    def prewitt(self, key='curr'):
        "Apply prewitt edge detection routine on a single channel image"
        # validate curr image
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        self.imgs[key] = ski.filters.prewitt(self.imgs['curr']).astype(float)
        return self.imgs[key]


if __name__ == '__main__':

    # examples
    IP_obj = Image_processor(path='/content/images/example_white.tif')

    # do a blured laplacian
    IP_obj.reset()
    IP_obj.gauss(sigma=1)
    IP_obj.gauss(sigma=1)
    IP_obj.laplace(size=3)
    IP_obj.grey_scale('curr')
    IP_obj.plot()

    # do an edge detection
    IP_obj.reset()
    IP_obj.grey_scale()
    IP_obj.prewitt()
    IP_obj.plot()
