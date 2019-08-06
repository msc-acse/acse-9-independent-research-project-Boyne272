# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:27:49 2019

@author: Richard Bonye
"""

import unittest
import torch
import numpy as np
from kmeans_basic import kmeans
from tools import set_seed, get_img

class Test_kmeans_basic(unittest.TestCase):

    def test_combined(self):
    
        set_seed(10)

        N = 1000
        x = torch.randn(N, 2, dtype=torch.float64)
        obj = kmeans(x, 10)

        assert len(obj.clusters) == 10, "there must be 10 clusters"

        obj.update_clusters()
        obj.update_centroids()
        obj.update_clusters()

        assert all([len(o) for o in obj.clusters.values()]), "No cluster should be empty after 1 step"

        for _ in range(10):
            obj.update_centroids()
            obj.update_clusters()

        cluster_sizes = [len(o) for o in obj.clusters.values()]
        assert min(cluster_sizes) > 30 and max(cluster_sizes) < 200, "in the case of normally distributed data there should not be clusters with a disproportionate number of vectors"

        
if __name__ == '__main__':
    unittest.main()