import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tools import percent_print
from time import clock


#%%


class kmeans_img():
    """
    
    """
    
    def __init__(self, img, nk, dist_func=None, args=None):
        
        """
        Parameters
        ----------
        
        img : 3d numpy array
            Image to be segmented by kmeans,
            expected in shape [x_dim, y_dim, rgb] with all values in the
            interval [0, 1], not 255.
            
        nk : int
            Number partitions (i.e. centroids)
            
        dist_func : callable, optional
            Method of calculating distance between two vectors,
            should accepts agruments f(vec1, vec2, args)
            
        args : tuple, optional
            Additional arguments to be passed to dist_func
        """
        
        # vectors
        self.x_dim, self.y_dim = img.shape[:2]
        self.n_pixels = self.x_dim * self.y_dim
        self.vectors = self.img_to_vectors(img)
        
        # setup centroids
        self.N = nk
        self.clusters = dict([(i, []) for i in range(self.N)])
        self.initalise_centroids()

        # set the distance function
        self.args = args
        if dist_func:
            self.distance = dist_func
        else:
            self.distance = lambda cent, vecs: (cent-vecs).norm(dim=1)
            
        # set the working tensors for distance calculations
        self.dist_array = torch.zeros([self.n_pixels, self.N])
        self.vector_cluster_id = torch.zeros([self.n_pixels])
        
    
    def img_to_vectors(self, img):
        """
        Convert 3d image array (x, y, rgb) into an array of 5d vectors
        (x, y, r, g, b)"
        """
        vecs = np.zeros([self.n_pixels, 5])
        x, y = range(self.x_dim), range(self.y_dim)
        X, Y = np.meshgrid(x, y)
        vecs[:, 0] = X.ravel()
        vecs[:, 1] = Y.ravel()
        vecs[:, 2] = img[:, :, 0].ravel()
        vecs[:, 3] = img[:, :, 1].ravel()
        vecs[:, 4] = img[:, :, 2].ravel()
        
        return torch.from_numpy(vecs).float()
    
    
    def initalise_centroids(self):
        """
        Initialise centroids on nk random points so they are not empty
        """
        indexs = torch.randint(0, self.n_pixels, [self.N]) ####################
#         print(indexs)
        self.centroids = self.vectors[indexs]
        
        
    def update_clusters(self):
        """
        The ###################################
        """
        
        # find the distance of each vector to eacg cluster
        # (memory intensive but much faster in implentation)
        for i in range(self.N):
            self.dist_array[:, i] = self.distance(self.centroids[i], self.vectors)
        
        # find the minimum distances
        self.vector_cluster_ids = torch.argmin(self.dist_array, dim = 1)
        
        for i in range(self.N):
            self.clusters[i] = (self.vector_cluster_ids==i).nonzero().squeeze()
    
    
    def update_centroids(self):
        """
        Move each centroid to the mean of its constituant vectors
        """
        for cent, vec_ids in self.clusters.items():
            if vec_ids.numel() != 0:
                self.centroids[cent] = self.vectors[vec_ids].mean(dim=0)
            else:
                self.centroids[cent] = self.vectors[torch.randint(self.n_pixels, [1])] ########################
        
        
    def iterate(self, n=1):
        """
        Loop the vector reassignment and the cluster center recaculation n
        times with a progress bar and a timer.
        """
        t = clock()
        for i in range(n):
            self.update_clusters()
            self.update_centroids()
            percent_print(i, n)
        print(" ", clock() - t, "s")


    def get_mask(self, edge=False, rgba=False):
        
        mask = self.vector_cluster_ids.view(self.x_dim, self.y_dim).numpy()
        
        if edge:
            laplacian = np.ones([3, 3])
            laplacian[1, 1] = -8
            mask = convolve2d(mask, laplacian, mode='full')
            mask = (mask > 0).astype(float)
            
            if rgba:
                zeros = np.zeros_like(mask)
                mask = np.dstack([mask, zeros, zeros, mask])
        
        return mask 
		
		
		
if __name__ == '__main__':

	from tools import get_img
	from kmeans_img import kmeans_img 
	set_seed(10)

	img = get_img("images/TX1_white_cropped.tif")
	obj = kmeans_img(img, 20, dist_func=dat_distance_metric)
	obj.iterate(10)

	mask = obj.get_mask(edge=True, rgba=True)
	fig, ax = plt.subplots(figsize=[20, 20])
	ax.imshow(img)
	ax.imshow(mask)
