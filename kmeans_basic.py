# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:08:17 2019

@author: Richard Bonye
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tools import percent_print


#%%

class kmeans():
    
    def __init__(self, vectors, number_centroids):
        
        # store vectors
        self.vectors = vectors.float()
        self.ndim = vectors.shape[-1]
        self.mean = self.vectors.mean(dim=0)
        self.std = self.vectors.std(dim=0)
        
        # setup centroids
        self.N = number_centroids
        self.clusters = dict([(i, []) for i in range(self.N)])
        self.initalise_centroids()

    
    def initalise_centroids(self):
        self.centroids = (torch.randn(self.N, self.ndim) + self.mean) * self.std
        
        
    def distance(self, a, b):
        return (a-b).norm()
        
        
    def update_clusters(self):
        
        # clear the old clusters
        self.clusters = dict([(i, []) for i in range(self.N)])
        
        # for every vector
        for vec_id, vecs in enumerate(self.vectors):
            min_dist = torch.Tensor([float("Inf")])
            min_cent = -1
            
            # find the min centroid distance
            for cent_id, cent in enumerate(self.centroids):
                dist = self.distance(cent, vecs)
                if dist < min_dist:
                    min_dist = dist
                    min_cent = cent_id
            
            self.clusters[min_cent].append(vec_id)
    
    
    def update_centroids(self):
        "mean method"
        for cent, vec_ids in self.clusters.items():
            if vec_ids != []:
                self.centroids[cent] = torch.Tensor(self.vectors[vec_ids]).mean(dim=0)
            else:
                self.centroids[cent] = (torch.randn(1, self.ndim) + self.mean) * self.std
        
        
    def iterate(self, n=1):
        for i in range(n):
            self.update_clusters()
            self.update_centroids()
            percent_print(i, n)
            
            
    def plot(self, ax=None):
        
        if not ax:
            fig, ax = plt.subplots(figsize=[10,10])
            
        for cent, cluster in self.clusters.items():
            ax.scatter(self.vectors[:, 0][cluster], self.vectors[:, 1][cluster])
            ax.plot(self.centroids[:, 0][cent], self.centroids[:, 1][cent], 'k*', ms=20)
            
        return ax
            
            
#%%

if __name__ == '__main__':
    
    # generate random 2d data
    N = 10000
    x = torch.randn(N, 2, dtype=torch.float64)
    
    # create the kmeans object
    obj = kmeans(x, 20)
    obj.update_clusters()
    obj.plot().set(title='initial')
    
    obj.iterate(10)
    obj.plot()
    
            
            