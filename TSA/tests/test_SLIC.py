# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Wed Aug 28 08:53:10 2019
"""


import unittest
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from TSA.kmeans import SLIC
from TSA.tools import set_seed, get_img


class Test_SLIC(unittest.TestCase):
    """
    Extensive set of tests for the SLIC class
    """

    dir_path = os.path.dirname(os.path.abspath(__file__))
    test_img_path_1 = dir_path + '/example_white.tif'


    def test_img_setup(self):
        """
        unit test that the img is converetd to vectors correctly and
        that the adjasent bins list makes sense
        """
        set_seed(10)
        img = get_img(self.test_img_path_1)
        obj = SLIC(img, [4, 4])

        # test the adjecent bins list
        assert len(obj._adj_bins) == 16, "should be 16 cells"

        assert len(obj._adj_bins[0]) == 4, "corner cell should only have 3 neighbours and itself"

        assert len(obj._adj_bins[1]) == 6, "edge cell should only have 5 neighbours and itself"

        # test the vectors are reasonable
        assert obj.vectors.shape[-1] == 5, "vectors must be 5d"

        assert obj.vectors.numel() == int(img.size * 5/3), \
            "There should be the same number of elements plus the " +\
            "x and y cordinates"

        # note the * here acts as an and opterator
        tmp = ((obj.vectors[:, 2] <= 1) * (obj.vectors[:, 2] >= 0) *
               (obj.vectors[:, 3] <= 1) * (obj.vectors[:, 3] >= 0) *
               (obj.vectors[:, 4] <= 1) * (obj.vectors[:, 4] >= 0))
        assert tmp.all(), "all pixel values must be in the range [0, 1]"


    def test_bin_setup(self):
        """
        unit test that the vectors have been binned correctly and
        clusters initalised correctly
        """
        img = get_img(self.test_img_path_1)
        obj = SLIC(img, [4, 4])

        # test bins
        assert sum([o.numel() for o in obj._vec_bins]) == obj._np, \
            "binned vectors needs to be same as number of pixels"

        assert obj._vec_bins.unique().numel() == 16, \
            "should be exactly 16 bins to which vectors belong"

        # test clusters
        assert sum([o.numel() for o in obj.vec_clusts]) == obj._np, \
            "clustered vectors needs to be same as number of pixels"

        assert obj.vec_clusts.unique().numel() == 16, \
            "should be exactly 16 clusters to which vectors belong"


    def test_one_iteration(self):
        """
        unit test that distance calculations, clusters and centroids still
        make sense after each of the three processes that make up a
        single iteration.
        """
        img = get_img(self.test_img_path_1)
        obj = SLIC(img, [4, 4])

        # 1 of 3 iteration parts
        obj._update_distances()

        # test distance list makes sense
        assert len(obj.distances) == 16, "should be 16 bin distance tensors"

        assert obj.distances[0].dtype == torch.float32, \
            "vc_dists should be a tensor with float values"

        assert all([a.shape[0] == b.shape[0] for a, b in
                    zip(obj.distances, obj._cluster_contense)]), \
            "The number of distance measures in each bin must match the " +\
            "number of vectors in that bin"

        assert all([a.shape[1] == b.shape[0] for a, b in
                    zip(obj.distances, obj._adj_bins)]), \
            "The number of distance measures per vector must match the " +\
            "number of adjasent centers for that bin"

        # 2 of 3 iteration parts
        obj._update_clusters()

        # test clusters make sense
        assert sum([o.numel() for o in obj._cluster_contense]) == obj._np, \
            "clustered vectors needs to be same as number of pixels"

        assert obj.vec_clusts.unique().numel() == 16, \
            "should be exactly 16 clusters"

        # 3 of 3 iteration parts
        obj._update_centroids()

        # test the new cetroids are still inside the image domain
        # note the * here acts as an and opterator between two binary arrays
        tmp = ((obj.centroids[:, 0] <= obj._dim_x) *
               (obj.centroids[:, 0] >= 0) *
               (obj.centroids[:, 1] <= obj._dim_y) *
               (obj.centroids[:, 1] >= 0) *
               (obj.centroids[:, 2] <= 1) *
               (obj.centroids[:, 2] >= 0) *
               (obj.centroids[:, 3] <= 1) *
               (obj.centroids[:, 3] >= 0) *
               (obj.centroids[:, 4] <= 1) *
               (obj.centroids[:, 4] >= 0))
        assert tmp.all(), "all pixel values must be in the image domain"


    def test_plot(self):
        """
        unit test that the iterate function, mask output and all the plot
        options run without raising any errors
        """
        img = get_img(self.test_img_path_1)
        img = img[:400, :400, :] # make smaller for faster tests
        obj = SLIC(img, [4, 4])

        # check the iterate command runs
        obj.iterate(2)

        # check every plot command runs
        for opt in ['default', 'setup', 'edges', 'img', 'centers',
                    'bins', 'time', 'bin_edges', 'segments']:
            obj.plot(opt)
            plt.close('all')

        # test the output mask
        assert obj.get_segmentation().shape == img.shape[:2], \
            "Mask must have same x,y dimensions as original image"



    def test_result(self):
        """
        Here an image is iterated to produce a mask (any comlex mask works).
        This mask is then coverted into a 3 channel image as used for a new
        segmentation task. This image has uniform regions with clear boundaries
        thus it should be easy to segment, and the segmentation should be the
        same as the original mask. There is some difference due to the large
        scale of the image and nature of the clusting, hence 80% agreement is
        requiered.
        """

        # setup object and iterate
        img_1 = get_img(self.test_img_path_1)
        img_1 = img_1[200:600, 200:600, :] # make smaller for faster test
        obj_1 = SLIC(img_1, [4, 4])
        obj_1.iterate(5)

        # extract mask
        mask_1 = obj_1.get_segmentation()

        # create an image from this mask
        img_2 = np.dstack([mask_1, mask_1, mask_1])

        # iterate with the mask image
        obj_2 = SLIC(img_2, [4, 4])
        obj_2 .iterate(10)

        mask_2 = obj_2.get_segmentation()

        similarity = np.isclose(mask_1, mask_2).mean()
        assert similarity > 0.8, 'masks should be at least 80% similar'


if __name__ == '__main__':

    # run all the tests if this is script is run independently
    test_obj = Test_SLIC()
    test_obj.test_img_setup()
    test_obj.test_bin_setup()
    test_obj.test_one_iteration()
    test_obj.test_plot()
    test_obj.test_result()
    print('\nall tests passed')

#     # unittest.main does not work in google colab, but should work elsewhere
#     unittest.main()
