import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tools import set_seed, get_img
from time import clock

#%%

class kmeans_local():
    
    def __init__(self, img, bin_grid=[1,1], dist_func='norm', device="cuda",
                 **kwargs):
        
        """
        Parameters
        ----------
        
        img : 3d numpy array
            Image to be segmented by kmeans,
            expected in shape [x_dim, y_dim, rgb] with all values in the
            interval [0, 1], not 255.
            
        bin_grid : tuple
            Length of 2, gives the number of initial partitions in x and y
            respectivly, therefore the number of segments is the product of
            these. Note that this intial segmentation also restrains the kmeans
            centers in space, forcing locality of segments and speeding up the
            algorithm.
            
        dist_func : callable, optional
            Method of calculating distance between two vectors,
            should accepts agruments f(vec1, vec2) 
            
        """
        
        # store given parameters
        self.device = device
        
        self.nx, self.ny = bin_grid 
        self.N = self.nx * self.ny
        
        self.y_dim, self.x_dim = img.shape[:2]
        self.n_pixels = self.x_dim * self.y_dim
        
        self.dx = self.x_dim / self.nx 
        self.dy = self.y_dim / self.ny
        assert self.dx == int(self.dx), "Must be evenly divisible in the x axis"
        assert self.dy == int(self.dy), "Must be evenly divisible in the y axis"

        
        # make the distance function
        if dist_func == 'custom':
            assert 'f' in kwards.keys(), 'f kwarg must be given'
            assert callable(kwargs['f']), 'f must be a function'
            self.distance = kwargs['f']
            
        elif dist_func == 'norm':
            self.distance = lambda cent, vecs: (cent-vecs).norm(dim=1)
            
        elif dist_func == 'scaled':
            assert 'scale' in kwards.keys(), 'scale kwarg must be given'
            self.distance = lambda cent, vecs: (
                (cent[2:]-vecs[:, 2:]).norm(dim=1) + # color distance
                (cent[:2]-vecs[:, :2]).norm(dim=1) * # position distance
                kwards['scale'] / np.sqrt(self.n_pixels/self.N)
            )
        
        # create the vectors
        self.vectors = self.img_to_vectors(img)  
        
        # create the pixel binning
        # vector bins is the bin each vector belongs to (fixed)
        # bin dict gives the neighbours of each bin
        self.bin_dict = self.create_bins()
        self.vector_bins = self.bin_vectors(self.vectors)
        
        # initalise centroids
        # vector bins is the cluster each vector belongs to (changes)
        # clusters is a dictionary of each clusters consitent vectors
        # centroids is an array of centroid centers
        self.vector_clusters = self.vector_bins.clone()
        self.clusters = dict([(i, (self.vector_clusters==i).nonzero().squeeze())
                              for i in range(self.N)])
        self.centroids = torch.empty([self.N, 5]).to(self.device)
        self.update_centroids()
        
        self.vector_bins = self.vector_bins.int().cpu().numpy() ###########################
            
    
    def img_to_vectors(self, img):
        """
        Convert 3d image array (x, y, rgb) into an array of 5d vectors
        (x, y, r, g, b). No scaling occurse here.
        """
        
        # create the 5d vector and x,y cordinates
        vecs = np.zeros([self.n_pixels, 5])
        x, y = range(self.x_dim), range(self.y_dim)
        X, Y = np.meshgrid(x, y)
        
        # set the position values
        vecs[:, 0] = X.ravel()
        vecs[:, 1] = Y.ravel()
        
        # set the color values
        vecs[:, 2] = img[:, :, 0].ravel()
        vecs[:, 3] = img[:, :, 1].ravel()
        vecs[:, 4] = img[:, :, 2].ravel()

        return torch.from_numpy(vecs).float().to(self.device)
    
                
    def bin_vectors(self, vecs, ret_dict=False):
        """
        Bin the vectors into the given bin grid
        If ret_dict this will return a dictionary of {bin: [vector indexs], ...}
        """
        
        # find the bin each pixel belongs to
        x_bins = (vecs[:, 0] / self.dx).int()
        y_bins = (vecs[:, 1] / self.dy).int()
        output = y_bins * self.nx + x_bins
        
        if ret_dict:
            output = dict([(i, (output==i).nonzero().squeeze())
                           for i in range(self.N)])
            
        return output
        
    
    def create_bins(self):
        "create a dictionary of which bins are adjasent"
                
        # create a dictionary of which bins are adjasent
        bin_dict = {}
        for i in range(self.N):
            x_cord, y_cord = i%self.nx, int(i/self.nx)
            cordinates = self.neighbours(x_cord, y_cord, self.nx, self.ny)
            indexs = [ y_ * self.nx + x_ for x_,y_ in cordinates ]
            bin_dict[i] = indexs
            
        return bin_dict
    
    
    def neighbours(self, x, y, x_max, y_max, r=1):
        """
        Find the neighbours in radius r to the x and y cordinate
        """
        return [(x_, y_) for x_ in range(x-r, x+r+1)
                         for y_ in range(y-r, y+r+1)
                         if (#(x != x_ or y != y_) and   # not the center
                             (0 <= x_ < x_max) and # not outside x range
                             (0 <= y_ < y_max))]   # not outside y range
        
    
    def update_centroids(self):
        """
        Move each centroid to the mean of its constituant vectors
        """
        for cent, vec_ids in self.clusters.items():

            if vec_ids.numel() == 0:
                raise(ValueError) # no cluster should ever be empty

            self.centroids[cent] = self.vectors[vec_ids].mean(dim=0)

            
    def update_clusters(self):
        """
        For every vector find the nearest cluster center from the adjasent bins 
        """
        
        centers_bin = self.bin_vectors(self.centroids, ret_dict=True)
        
        for vec_id, bin_no in enumerate(self.vector_bins):
            bins_to_search = self.bin_dict[bin_no]
            
            centroid_ids = [centers_bin[c] for c in bins_to_search]
            nearby_centroids = self.centroids[centroid_ids, :]
            
            distances = self.distance(self.vectors[vec_id], nearby_centroids)
            
            self.vector_clusters[vec_id] = centroid_ids[torch.argmin(distances)]
                           
#         # find the distance of each vector to eacg cluster
#         # (memory intensive but much faster in implentation)
#         for i in range(self.N):
#             self.dist_array[:, i] = self.distance(self.centroids[i], self.vectors)
        
#         # find the minimum distances
#         self.vector_cluster_ids = torch.argmin(self.dist_array, dim = 1)
        
#         for i in range(self.N):
#             self.clusters[i] = (self.vector_cluster_ids==i).nonzero().squeeze()
            
    
    def get_mask(self, option, edges=True, rgba=True):
        
        if option == 'bins':
            mask = self.vector_bins
            
        elif option == 'seg':
            mask = self.vector_clusters
        
        else:
            raise(ValueError) # option not recognised
    
        # reshape the mask and make it into numpy
        mask = mask.view(self.y_dim, self.x_dim).cpu().numpy()
    
        # if only the edges are wanted use a laplacian convolution
        if edges:
            laplacian = np.ones([3, 3])
            laplacian[1, 1] = -8
            mask = convolve2d(mask, laplacian, mode='valid')
            mask = (mask > 0).astype(float)
            
            # if wanted make into rgba form so that it can overlay in imshow
            if rgba:
                zeros = np.zeros_like(mask)
                mask = np.dstack([mask, zeros, zeros, mask])
        
        return mask
		
#%%

if __name__ == '__main__':

	set_seed(10)
	img = get_img("images/TX1_polarised_cropped.tif")
	obj = kmeans_local(img, [4,4])