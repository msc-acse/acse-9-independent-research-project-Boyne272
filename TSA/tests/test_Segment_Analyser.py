# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Wed Aug 28 08:54:17 2019
"""


import unittest
import matplotlib.pyplot as plt
import numpy as np
from TSA.pre_post_processing import Segment_Analyser


class Test_Segment_Analyser(unittest.TestCase):
    """
    Some basic tests for Segment_Analyser. It is very hard to test
    the user interaction part of this code, so this is not verified here.
    All property calculations are verified against known values for a dummy
    segmentation and clustering.
    """


    def test_full(self):
        """
        Creates a non uniform dummy mask with a 50/50 clustering. Ensures the
        code runs and plots without issue. Compares all properties to those
        expected from runs were it was known to be working (by inspection
        of the grid).
        """

        # create a dummy mask
        mask_upper = np.array([[0, 0, 0, 0],
                               [1, 1, 2, 2],
                               [3, 3, 4, 5],
                               [6, 7, 8, 9]]).repeat(5, axis=0).repeat(10, axis=1)
        mask_lower = np.array([[10, 11],
                               [12, 13]]).repeat(10, axis=0).repeat(20, axis=1)
        mask = np.vstack((mask_upper, mask_lower))

        # create a dummy clustering
        cluster = np.zeros_like(mask)
        cluster[20:, :] = 1

        # create the analysis object
        example_obj = Segment_Analyser(mask, mask, cluster)
        example_obj.labels = {'upper':0, 'lower':1} # set the labels manually

        # plot the generated segments and clusters
        example_obj.plot_cluster('upper')
        example_obj.plot_cluster('lower')

        # observe the different properties
        properties = [example_obj.get_composition(return_arr=True),
                      example_obj.get_grain_count(return_arr=True),
                      example_obj._get_span('upper', return_arr=True),
                      example_obj.get_gsd('upper', return_arr=True,
                                          span=False),
                      example_obj._get_span('lower', return_arr=True),
                      example_obj.get_gsd('lower', return_arr=True,
                                          span=False)]

        # close the many figures just opened
        plt.close('all')

        # write the expected outcomes (some of these can be seen to be true
        # by inspection, others require a bit of calculation to confirm)
        expected = [np.array([0.5, 0.5]),
                    np.array([10, 4]),
                    np.array([39.0, 19.4164878389476, 19.4164878389476,
                              19.4164878389476, 9.848857801796104, 9.848857801796104,
                              9.848857801796104, 9.848857801796104, 9.848857801796104,
                              9.848857801796104]),
                    np.array([[200.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0,
                               50.0, 50.0],
                              [40.0, 43.0, 43.0, 43.0, 26.0, 23.0, 23.0, 26.0,
                               26.0, 23.0],
                              [5.0, 2.3255813953488373, 2.3255813953488373,
                               2.3255813953488373, 1.9230769230769231,
                               2.1739130434782608, 2.1739130434782608,
                               1.9230769230769231, 1.9230769230769231,
                               2.1739130434782608]]),
                    np.array([21.02379604162864, 21.02379604162864,
                              21.02379604162864, 21.02379604162864]),
                    np.array([[200.0, 200.0, 200.0, 200.0],
                              [48.0, 48.0, 29.0, 29.0],
                              [4.166666666666667, 4.166666666666667,
                               6.896551724137931, 6.896551724137931]])]
        for i, (p, e) in enumerate(zip(properties, expected)):
            assert (p == e).all(), 'property %i was not as expected'%i


if __name__ == '__main__':

    # run all the tests if this is script is run independently
    test_obj = Test_Segment_Analyser()
    test_obj.test_full()
    print('\n all tests passed')

#     # unittest.main does not work in google colab, but should work elsewhere
#     unittest.main()
