# -*- coding: utf-8 -*-
"""
@author: Richard Bonye
"""

import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt

# from SLIC import SLIC
from tools import set_seed, get_img


class Test_SLIC(unittest.TestCase):

    
    def test_img_setup(self):
        """
        unit test that the img is converetd to vectors correctly and
        that the adjasent bins list makes sense
        """
        set_seed(10)
        img = get_img("images/TX1_white_cropped.tif")
        obj = SLIC(img, [4,4])
        
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
        img = get_img("images/TX1_white_cropped.tif")
        obj = SLIC(img, [4,4])

        # test bins
        assert sum([o.numel() for o in obj._vec_bins]) == obj._Np, \
            "binned vectors needs to be same as number of pixels"
        
        assert obj._vec_bins.unique().numel() == 16, \
            "should be exactly 16 bins to which vectors belong"

        # test clusters
        assert sum([o.numel() for o in obj.vec_clusts]) == obj._Np, \
            "clustered vectors needs to be same as number of pixels"
        
        assert obj.vec_clusts.unique().numel() == 16, \
            "should be exactly 16 clusters to which vectors belong"

    
    def test_one_iteration(self):
        """
        unit test that distance calculations, clusters and centroids still 
        make sense after each of the three processes that make up a
        single iteration.
        """
        img = get_img("images/TX1_white_cropped.tif")
        obj = SLIC(img, [4,4])

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
        assert sum([o.numel() for o in obj._cluster_contense]) == obj._Np, \
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
        img = get_img("images/TX1_white_cropped.tif")
        obj = SLIC(img, [4,4])

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
        acceptance test that the outcome is as expected for a simple image by
        checking a handful of manually identified points are seperated by the
        algorithm
        """
        # setup and iterate
        img = get_img("/content/images/SM1_250_250.tif")
        obj = SLIC(img, [5,5])
        obj.iterate(5)
        
        # extract mask
        mask = obj.get_segmentation()
        
        # define known points
        known_points = [[10, 10],
                        [50, 10],
                        [10, 50],
                        [240, 10],
                        [240, 50]]
        
        # for each pair check they are in different segments
        for x1, y1 in known_points:
            for x2, y2 in known_points:
                if x1 != x2 and y1 != y2:
                    assert mask[y1, x1] != mask[y2, x2], \
                        "Mask does not agree with expected"

        
if __name__ == '__main__':
    
    # run all the tests if this is script is run independently
    test_obj = Test_SLIC()
    test_obj.test_img_setup()
    test_obj.test_bin_setup()
    test_obj.test_one_iteration()
    test_obj.test_plot()
    test_obj.test_result()
    
#     # unittest.main does not work in google colab, but should work elsewhere
#     unittest.main()