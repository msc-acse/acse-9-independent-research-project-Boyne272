# -*- coding: utf-8 -*-
"""
@author: Richard Bonye
"""

import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt

from SLIC import SLIC
from tools import set_seed, get_img


class Test_SLIC(unittest.TestCase):

    
    def test_img_setup(self):
        """
        test that the img is converetd to vectors correctly and
        that the adjasent bins list makes sense
        """
        set_seed(10)
        img = get_img("images/TX1_white_cropped.tif")
        obj = SLIC(img, [4,4])
        
        # test the adjecent bins list
        assert len(obj.adj_bins) == 16, "should be 16 cells"
        assert len(obj.adj_bins[0]) == 4, "corner cell should only have 3 neighbours and itself"
        assert len(obj.adj_bins[1]) == 6, "edge cell should only have 5 neighbours and itself"

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
        test that the vectors have been binned correctly and
        clusters initalised correctly
        """
        img = get_img("images/TX1_white_cropped.tif")
        obj = SLIC(img, [4,4])

        # test bins
        assert sum([o.numel() for o in obj.vec_bins_list]) == obj.Np, \
            "binned vectors needs to be same as number of pixels"
        assert obj.vec_bins_tensor.unique().numel() == 16, \
            "should be exactly 16 bins to which vectors belong"

        # test clusters
        assert sum([o.numel() for o in obj.cluster_list]) == obj.Np, \
            "clustered vectors needs to be same as number of pixels"
        assert obj.cluster_tensor.unique().numel() == 16, \
            "should be exactly 16 clusters to which vectors belong"

    
    def test_one_iteration(self):
        """
        Test that distance calculations, clusters and centroids still 
        make sense after a single iteration.
        """
        img = get_img("images/TX1_white_cropped.tif")
        obj = SLIC(img, [4,4])

        obj.update_distances()
        
        # test distance list makes sense
        assert len(obj.vc_dists) == 16, "should be 16 bin distance tensors"
        assert obj.vc_dists[0].dtype == torch.float32, \
            "vc_dists should be a tensor with float values"
        assert all([a.shape[0] == b.shape[0] for a, b in 
                    zip(obj.vc_dists, obj.vec_bins_list)]), \
            "The number of distance measures in each bin must match the " +\
            "number of vectors in that bin"
        assert all([a.shape[1] == b.shape[0] for a, b in 
                    zip(obj.vc_dists, obj.adj_bins)]), \
            "The number of distance measures per vector must match the " +\
            "number of adjasent centers for that bin"
        
        obj.update_clusters()

        # test clusters make sense
        assert sum([o.numel() for o in obj.cluster_list]) == obj.Np, \
            "clustered vectors needs to be same as number of pixels"
        assert obj.cluster_tensor.unique().numel() == 16, \
            "should be exactly 16 clusters" 
    
        obj.update_centroids()
        
        # test the new cetroids are still inside the image domain
        # note the * here acts as an and opterator
        tmp = ((obj.centroids[:, 2] <= 1) * (obj.centroids[:, 2] >= 0) *
               (obj.centroids[:, 2] <= 1) * (obj.centroids[:, 2] >= 0) *
               (obj.centroids[:, 2] <= 1) * (obj.centroids[:, 2] >= 0) *
               (obj.centroids[:, 3] <= 1) * (obj.centroids[:, 3] >= 0) *
               (obj.centroids[:, 4] <= 1) * (obj.centroids[:, 4] >= 0))
        assert tmp.all(), "all pixel values must be in the range [0, 1]"
    
    
    def test_plot(self):
        """
        check that the iterate function and all the plot options run
        (Not perfect but better than nothing)
        """
        img = get_img("images/TX1_white_cropped.tif")
        obj = SLIC(img, [4,4])

        obj.iterate(1)

        for opt in ['default', 'setup', 'edges', 'img', 'centers', 
                    'bins', 'time', 'bin_edges', 'segments', 'setup']:
            obj.plot(opt)
    
        
if __name__ == '__main__':
    unittest.main()