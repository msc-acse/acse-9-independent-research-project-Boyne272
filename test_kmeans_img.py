# -*- coding: utf-8 -*-
"""

@author: Richard Bonye
"""

import unittest
import torch
import numpy as np
from kmeans_img import kmeans_img
from tools import set_seed, get_img

class Test_kmeans_img(unittest.TestCase):

    def test_clusters(self):
        "test clusters are initalised and iterate without any empty clusters"
    
        set_seed(10)
    
        img = get_img("images/TX1_polarised_cropped.tif")
        obj = kmeans_img(img, 10)
        obj.update_clusters()

        assert all([len(o) for o in obj.clusters.values()]), "no cluster should be empty intially"

        obj.update_centroids()
        obj.update_clusters()

        assert all([len(o) for o in obj.clusters.values()]), "no cluster should be after 1 iteration"
        
    
    def test_vector_initalisation(self):
        "test the convergence from 2d rgb image to 5d vectors is correct"
        
        set_seed(10)
        img = get_img("images/TX1_polarised_cropped.tif")
        obj = kmeans_img(img, 10)

        assert obj.vectors.numel() == int(img.size * 5/3), "There should be the same number of elements plus the x and y cordinates"
        tmp = ((obj.vectors[:, 2] <= 1) * (obj.vectors[:, 2] >= 0) *
               (obj.vectors[:, 3] <= 1) * (obj.vectors[:, 3] >= 0) *
               (obj.vectors[:, 4] <= 1) * (obj.vectors[:, 4] >= 0))
        assert tmp.all(), "all pixel values must be in the range [0, 1]"
        
if __name__ == '__main__':
    unittest.main()