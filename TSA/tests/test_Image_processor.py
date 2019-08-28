# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Wed Aug 28 08:49:18 2019
"""


import unittest
import os
import matplotlib.pyplot as plt
import numpy as np
from TSA.tools import get_img
from TSA.pre_post_processing import Image_processor


class Test_Image_processor(unittest.TestCase):
    """
    This module is a wrapper around skimage for convinence, hence the tests
    are relativly short.
    """

    test_img_path = os.path.dirname(os.path.abspath(__file__)) + \
        '/example_white.tif'

    def test_initalisation(self):
        """
        test initalisation and assoiated assertions behave as expected
        """

        path = self.test_img_path

        # test initalisation with path
        obj = Image_processor(path=path)
        assert len(obj.imgs) == 2, 'should have initlaised 2 imgs in dict'

        # test initalisation with array
        obj = Image_processor(img=get_img(path))
        assert len(obj.imgs) == 2, 'should have initlaised 2 imgs in dict'

        try:
            obj = Image_processor(img=get_img(path).tolist())
            raise ValueError # should have broken
        except AssertionError:
            pass

        try:
            obj = Image_processor(img=get_img(path).ravel())
            raise ValueError # should have broken
        except AssertionError:
            pass


    def test_chain_filters(self):
        """
        test multiple filters can be stacked together correctly
        """
        obj = Image_processor(self.test_img_path)

        gaussian = obj.gauss()
        sobel_1 = obj.sobel('sob')
        assert 'sob' in obj.imgs.keys(), 'image should have been saved as sob'

        gauss_obj = Image_processor(img=gaussian)
        sobel_2 = gauss_obj.sobel()

        assert np.isclose(sobel_1, sobel_2).all(), \
            'the same operations in different order should be the same'


    def test_utilities(self):
        """
        test the utility functions (save, rebase, etc) behave as expected.
        Save is not tested due to complications with removing the created image.
        """
        obj = Image_processor(self.test_img_path)
        obj.grey_scale()
        obj.reset()
        assert obj.imgs['curr'].ndim == 3, 'image should have been reset'

        # check plot runs
        obj.plot()
        obj.plot('original')
        plt.close('all')

        # check one can rebase with a new image
        obj.grey_scale()
        obj.rebase()
        assert obj.imgs['original'].ndim == 2, \
            'original image should have been changed'


    def test_processes(self):
        """
        test each image processing function runs without issue
        (since skimage is likly working correctly the outputs do not
        need to be validated)
        """

        obj_color = Image_processor(self.test_img_path)
        obj_grey = Image_processor(img=obj_color.grey_scale())

        # test each multi channel function runs
        obj_color.reset()
        obj_color.hsv()

        obj_color.reset()
        obj_color.select_channel(channel=0)

        obj_color.reset()
        obj_color.gauss(sigma=3)

        obj_color.reset()
        obj_color.sobel()

        obj_color.reset()
        obj_color.scharr()

        obj_color.reset()
        obj_color.laplace()

        obj_color.reset()
        obj_color.hog()


        # test each single channel function runs
        obj_grey.reset()
        obj_grey.normalise()

        obj_grey.reset()
        obj_grey.dilation()

        obj_grey.reset()
        obj_grey.erosion()

        obj_grey.reset()
        obj_grey.median()

        obj_grey.reset()
        obj_grey.gabor_filters(frequency=2, n_angs=4)

        obj_grey.reset()
        obj_grey.gauss_grey()

        obj_grey.reset()
        obj_grey.lbp()

        obj_grey.reset()
        obj_grey.canny()

        obj_grey.reset()
        obj_grey.prewitt()

        obj_grey.reset()
        obj_grey.threshold(.5)

        obj_grey.reset()
        obj_grey.hog()



if __name__ == '__main__':

    # run all the tests if this is script is run independently
    test_obj = Test_Image_processor()
    test_obj.test_initalisation()
    test_obj.test_chain_filters()
    test_obj.test_utilities()
    test_obj.test_processes()
    print('all tests passed')

#     # unittest.main does not work in google colab, but should work elsewhere
#     unittest.main()
