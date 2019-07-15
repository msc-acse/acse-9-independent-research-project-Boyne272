# -*- coding: utf-8 -*-
"""
@author: Richard Bonye
"""

import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt

from MSLIC import MSLIC_wrapper
from tools import set_seed, get_img


class Test_MSLIC_wrapper(unittest.TestCase):

    
    def test_distance_combination(self):
        """
        test that the img is converetd to vectors correctly and
        that the adjasent bins list makes sense
        """
        # setup
        set_seed(10)
        img_w = get_img("images/TX1_white_cropped.tif")
        img_p = get_img("images/TX1_polarised_cropped.tif")
        obj = MSLIC_wrapper((img_w, img_p), [4,4])
        
        # find the combined distance
        for o in obj.SLIC_objs:
            o.update_distances()
        obj.combined_distance()
        
        # test the combined distance makes sense
        assert all([a.shape == b.shape for a,b in
                    zip(obj.combo_dist, obj.SLIC_objs[0].vc_dists)]),\
            "the combined distance must have the same shape as the " +\
            "distances for each SLIC object"
        assert all([a.shape == b.shape for a,b in
                    zip(obj.combo_dist, obj.SLIC_objs[1].vc_dists)]),\
            "the combined distance must have the same shape as the " +\
            "distances for each SLIC object"
        
        
    def test_assignment_to_SLIC(self):
        """
        test that MSLIC_wrapper is assigining the distances correctly
        and that the SLIC algorithms are continuing without error
        """
        # setup
        set_seed(10)
        img_w = get_img("images/TX1_white_cropped.tif")
        img_p = get_img("images/TX1_polarised_cropped.tif")
        obj = MSLIC_wrapper((img_w, img_p), [4,4])
        
        # iterate
        obj.iterate(1)
        
        # check both SLIC instances agree on the clustering
        assert all(obj.SLIC_objs[0].cluster_tensor == \
                   obj.SLIC_objs[1].cluster_tensor),\
            "both SLIC instances must agree on the clustering"
        
        # check the SLIC instances are still functional
        self.previous_SLIC(obj.SLIC_objs[0])
        self.previous_SLIC(obj.SLIC_objs[1])
        
    
    def previous_SLIC(self, obj):
        """
        use the SLIC tests after an iteration to ensure the assignment
        the MSLIC_wrapper does is not impeding its algorithm in any way
        """
        
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

        # test clusters make sense
        assert sum([o.numel() for o in obj.cluster_list]) == obj.Np, \
            "clustered vectors needs to be same as number of pixels"
        assert obj.cluster_tensor.unique().numel() == 16, \
            "should be exactly 16 clusters" 
    
        
        # test the new cetroids are still inside the image domain
        # note the * here acts as an and opterator
        tmp = ((obj.centroids[:, 0] <= obj.dim_x) * (obj.centroids[:, 0] >= 0) *
               (obj.centroids[:, 1] <= obj.dim_y) * (obj.centroids[:, 1] >= 0) *
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
        img_w = get_img("images/TX1_white_cropped.tif")
        img_p = get_img("images/TX1_polarised_cropped.tif")
        obj = MSLIC_wrapper((img_w, img_p), [4,4])
        
        # iterate
        obj.iterate(1)
        
        for opt in ['default', 'setup', 'edges', 'img', 'centers', 
                    'bins', 'time', 'bin_edges', 'segments', 'setup']:
            obj.plot([0,1], opt)
            plt.close('all')
        
        # check that plotting a just one instance also works
        obj.plot([0], 'default')
        obj.plot([1], 'default')
        
        
if __name__ == '__main__':
    unittest.main()