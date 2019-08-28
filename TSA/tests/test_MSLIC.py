# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Wed Aug 28 08:51:40 2019
"""


import unittest
import os
import torch
import matplotlib.pyplot as plt

from TSA.kmeans import MSLIC_wrapper
from TSA.tools import set_seed, get_img


class Test_MSLIC_wrapper(unittest.TestCase):
    """
    Tests for the MSLIC class. Note these are all unit tests as this is doing
    nothing new over the SLIC class (except for the distance combination
    which is tested below) hence results based tests are not needed
    """

    dir_path = os.path.dirname(os.path.abspath(__file__))
    test_img_path_1 = dir_path + '/example_white.tif'
    test_img_path_2 = dir_path + '/example_polar.tif'


    def test_distance_combination(self):
        """
        test that the img is converetd to vectors correctly and
        that the adjasent bins list makes sense
        """
        # setup
        set_seed(10)
        img_w = get_img(self.test_img_path_1)
        img_p = get_img(self.test_img_path_2)
        obj = MSLIC_wrapper((img_w, img_p), [4, 4])

        # find the combined distance
        for slic_obj in obj.SLIC_objs:
            slic_obj._update_distances()
        combo_dist = obj._combined_distance()

        # test the combined distance is the correct shape
        assert all([a.shape == b.shape for a, b in
                    zip(combo_dist, obj.SLIC_objs[0].distances)]), \
            "combined distance must have the same shape as the " +\
            "distances for each SLIC object"

        assert all([a.shape == b.shape for a, b in
                    zip(combo_dist, obj.SLIC_objs[1].distances)]), \
            "combined distance must have the same shape as the " +\
            "distances for each SLIC object"

        # check the combined distance is different from the indevidual ones
        for combo, dist1, dist2 in zip(combo_dist,
                                       obj.SLIC_objs[0].distances,
                                       obj.SLIC_objs[1].distances):
            assert not (combo == dist1).all(), \
                "combined distance must be different to indevidual distances"
            assert not (combo == dist2).all(), \
                "combined distance must be different to indevidual distances"


    def test_assignment_to_slic(self):
        """
        test that MSLIC_wrapper is assigining the distances correctly
        and that the SLIC algorithms are continuing without error
        """
        # setup
        set_seed(10)
        img_w = get_img(self.test_img_path_1)
        img_p = get_img(self.test_img_path_2)
        obj = MSLIC_wrapper((img_w, img_p), [4, 4])

        # iterate
        obj.iterate(1)

        # check both SLIC instances agree on the clustering
        assert all(obj.SLIC_objs[0].vec_clusts == \
                   obj.SLIC_objs[1].vec_clusts),\
            "both SLIC instances must agree on the clustering"

        # check the SLIC instances are still functional
        self.previous_slic(obj.SLIC_objs[0])
        self.previous_slic(obj.SLIC_objs[1])


    def previous_slic(self, obj):
        """
        use the SLIC tests after an iteration to ensure the assignment
        the MSLIC_wrapper does is not impeding its algorithm in any way
        """

        # test distance list makes sense
        assert len(obj.distances) == 16, "should be 16 bin distance tensors"

        assert obj.distances[0].dtype == torch.float32, \
            "vc_dists should be a tensor with float values"

        assert all([a.shape[0] == b.shape[0] for a, b in
                    zip(obj.distances, obj._bins_list)]), \
            "The number of distance measures in each bin must match the " +\
            "number of vectors in that bin"

        assert all([a.shape[1] == b.shape[0] for a, b in
                    zip(obj.distances, obj._adj_bins)]), \
            "The number of distance measures per vector must match the " +\
            "number of adjasent centers for that bin"

        # test clusters make sense
        assert sum([o.numel() for o in obj._cluster_contense]) == obj._np, \
            "clustered vectors needs to be same as number of pixels"

        assert obj.vec_clusts.unique().numel() == 16, \
            "should be exactly 16 clusters"

        # test the new cetroids are still inside the image domain
        # note the * here acts as an and opterator
        tmp = ((obj.centroids[:, 0] <= obj._dim_x) * (obj.centroids[:, 0] >= 0) *
               (obj.centroids[:, 1] <= obj._dim_y) * (obj.centroids[:, 1] >= 0) *
               (obj.centroids[:, 2] <= 1) * (obj.centroids[:, 2] >= 0) *
               (obj.centroids[:, 3] <= 1) * (obj.centroids[:, 3] >= 0) *
               (obj.centroids[:, 4] <= 1) * (obj.centroids[:, 4] >= 0))
        assert tmp.all(), "all pixel values must be in the image domian"


    def test_plot(self):
        """
        check that all the plot options run
        (Not perfect but better than nothing)
        """
        # setup
        set_seed(10)
        img_w = get_img(self.test_img_path_1)
        img_p = get_img(self.test_img_path_2)
        obj = MSLIC_wrapper((img_w, img_p), [4, 4])

        # iterate
        obj.iterate(2)

        for opt in ['default', 'setup', 'edges', 'img', 'centers',
                    'bins', 'time', 'bin_edges', 'segments', 'setup']:
            obj.plot([0, 1], opt)
            plt.close('all')

        # check that plotting a just one instance also works
        obj.plot([0], 'default')
        obj.plot([1], 'default')


if __name__ == '__main__':

    # run all the tests if this is script is run independently
    test_obj = Test_MSLIC_wrapper()
    test_obj.test_distance_combination()
    test_obj.test_assignment_to_slic()
    test_obj.test_plot()
    print('\n all tests passed')

#     # unittest.main does not work in google colab, but should work elsewhere
#     unittest.main()
