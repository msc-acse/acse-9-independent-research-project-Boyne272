# -*- coding: utf-8 -*-
"""
@author: Richard Bonye
"""

import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_local import kmeans_local
from tools import set_seed, get_img


class Test_kmeans_local(unittest.TestCase):

    
    def test_img_setup(self):
        "test that the img and vectors are made correctly"
        set_seed(10)
        img = get_img("images/TX1_white_cropped.tif")
        obj = kmeans_local(img, [4,4])

        assert len(obj.adj_bins) == 16, "should be 16 cells"
        assert len(obj.adj_bins[0]) == 4, "corner cell should only have 3 neighbours and itself"
        assert len(obj.adj_bins[1]) == 6, "edge cell should only have 5 neighbours and itself"

        assert obj.vectors.numel() == int(img.size * 5/3), "There should be the same number of elements plus the x and y cordinates"
        tmp = ((obj.vectors[:, 2] <= 1) * (obj.vectors[:, 2] >= 0) *
               (obj.vectors[:, 3] <= 1) * (obj.vectors[:, 3] >= 0) *
               (obj.vectors[:, 4] <= 1) * (obj.vectors[:, 4] >= 0))
        assert tmp.all(), "all pixel values must be in the range [0, 1]"
        return   
    
    
    def test_bin_setup(self):
        """
        test that the vectors have been binned correctly and
        clusters initalised correctly
        """
        img = get_img("images/TX1_white_cropped.tif")
        obj = kmeans_local(img, [4,4])

        # test bins
        assert sum([o.numel() for o in obj.vec_bins_list]) == obj.Np, \
            "binned vectors needs to be same as number of pixels"
        assert obj.vec_bins_tensor.unique().numel() == 16, \
            "should be exactly 16 bins"

        # test clusters
        assert sum([o.numel() for o in obj.cluster_list]) == obj.Np, \
            "clustered vectors needs to be same as number of pixels"
        assert obj.cluster_tensor.unique().numel() == 16, \
            "should be exactly 16 clusters"
        
        return
    
    
    def test_iteration(self):
        "test that clusters still make sense after a single iteration"
        img = get_img("images/TX1_white_cropped.tif")
        obj = kmeans_local(img, [4,4])

        obj.update_clusters()
        obj.update_centroids()

        assert sum([o.numel() for o in obj.cluster_list]) == obj.Np, \
            "clustered vectors needs to be same as number of pixels"
        assert obj.cluster_tensor.unique().numel() == 16, \
            "should be exactly 16 clusters"
        return    
    
    
    def test_plot(self):
        "check that all the plot code runs (not a full proof test but better than nothing)"
        img = get_img("images/TX1_white_cropped.tif")
        obj = kmeans_local(img, [4,4])

        obj.iterate(2)

        fig, ax = plt.subplots()
        for opt in ['default', 'edges', 'img', 'centers', 'bins']:
            obj.plot(opt, ax=ax)

        obj.plot('time')
        return
    
        
if __name__ == '__main__':
    unittest.main()