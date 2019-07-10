# -*- coding: utf-8 -*-
"""

@author: Richard Bonye
"""

import unittest
import torch
import numpy as np
from kmeans_local_failed import kmeans_local
from tools import set_seed, get_img

class Test_kmeans_local_failed(unittest.TestCase):

    def test_bins(self):
        # test the bin creation
        set_seed(10)
        img = get_img("images/TX1_polarised_cropped.tif")
        obj = kmeans_local(img, [4,4])

        assert len(obj.vector_bins.unique()) == 16, "should be 16 cells"
        assert len(obj.bin_dict) == 16, "should be 16 cells"
        assert len(obj.bin_dict[0]) == 4, "corner cell should only have 3 neighbours and itself"
        assert len(obj.bin_dict[1]) == 6, "edge cell should only have 5 neighbours and itself"
        
    
    def test_centroids(self):
        # test the centroids are initalised correctly
        set_seed(10)
        img = get_img("images/TX1_polarised_cropped.tif")
        obj = kmeans_local(img, [4,4])

        assert len(obj.clusters) == 16, "should be 16 clusters"
        assert all([len(c) > 0 for c in obj.clusters.values()]), "should be at more than one vector in each cluster"
        assert len(obj.vector_clusters.unique()) == 16, "There should be 16 clusters"
    
    
    def test_vector_initalisation(self):
        "test the convergence from 2d rgb image to 5d vectors is correct"
        
        # test the vector coversion
        set_seed(10)
        img = get_img("images/TX1_polarised_cropped.tif")
        obj = kmeans_local(img, [2, 2])

        assert obj.vectors.numel() == int(img.size * 5/3), "There should be the same number of elements plus the x and y cordinates"
        tmp = ((obj.vectors[:, 2] <= 1) * (obj.vectors[:, 2] >= 0) *
               (obj.vectors[:, 3] <= 1) * (obj.vectors[:, 3] >= 0) *
               (obj.vectors[:, 4] <= 1) * (obj.vectors[:, 4] >= 0))
        assert tmp.all(), "all pixel values must be in the range [0, 1]"
        
if __name__ == '__main__':
    unittest.main()