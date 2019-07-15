import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from tools import set_seed, get_img, progress_bar


class img_base():
    """
    The base needed to convert a image array into a 5d vector
    """
    
    def __init__(self, img, **kwargs):
        "setup the img and vectors"
        
        assert img.ndim == 3, "image must be a 3d array"
        assert img.shape[-1] == 3, "image must be of form (x ,y, rgb)"
        
        self.dim_y, self.dim_x = img.shape[:2]

        self.Np = self.dim_x * self.dim_y      # the number of pixels
        self.vectors = self.img_to_vectors(img)
        self.img = img
        
    
    def img_to_vectors(self, img):
        """
        Convert 3d image array (x, y, rgb) into an array of 5d vectors
        (x, y, r, g, b)"
        """
        
        # the x and y cordinates
        X, Y = np.meshgrid(range(self.dim_x),
                           range(self.dim_y))
        
        # create the 5d vectors
        vecs = np.zeros([self.Np, 5])
        vecs[:, 0] = X.ravel()
        vecs[:, 1] = Y.ravel()
        vecs[:, 2] = img[:, :, 0].ravel()
        vecs[:, 3] = img[:, :, 1].ravel()
        vecs[:, 4] = img[:, :, 2].ravel()
        
        return torch.from_numpy(vecs).float()
    
    
    def tensor_to_mask(self, tensor):
        "Reshape a tensor of values into the original image shape"
        return tensor.view(self.dim_y, self.dim_x).cpu().numpy()
    
    
    def mask_aplha_values(self, mask, color='r'):
        "Take a 2d mask and return a 3d rgba mask for imshow overlaying"
        
        assert type(mask) == type(np.array([])), "mask must be a numpy array"
        assert mask.ndim == 2, "mask must be a 2d array"
        
        zeros = np.zeros_like(mask)
        
        rgba = np.dstack([zeros, zeros, zeros, mask])
        
        if color == 'r':
            rgba[:, :, 0] = mask
        elif color == 'g':
            rgba[:, :, 1] = mask
        elif color == 'b':
            rgba[:, :, 2] = mask
            
        return rgba
    
    
    def outline(self, mask):
        """
        Take a 2d mask and use a laplacian convolution to find the segment 
        outlines
        """
        
        assert type(mask) == type(np.array([])), "mask must be a numpy array"
        assert mask.ndim == 2, "mask must be a 2d array"
        
        laplacian = np.ones([3, 3])
        laplacian[1, 1] = -8
        edges = sig.convolve2d(mask, laplacian, mode='valid')
        edges = (edges > 0).astype(float)
        
        return edges
        
        

class bin_base():
    """
    The base needed to bin vectors into a given grid
    """
    
    def __init__(self, bin_grid, **kwargs):
        "Setup the image grid with binned pixels into"
        
        assert len(bin_grid) == 2, "bin grid must be 2d"
        assert type(bin_grid[0]) == int, "grid must be integers"
        
        self.Nx, self.Ny = bin_grid # number of bin divisions in x and y
        self.Nk = self.Nx * self.Ny # number of k means centroids
        
        self.bin_dx = self.dim_x / self.Nx  # size of bins in x
        self.bin_dy = self.dim_y / self.Ny  # size of bins in y
        
        assert self.bin_dx == int(self.bin_dx), \
            "Must be evenly divisible in the x axis"
        assert self.bin_dy == int(self.bin_dy), \
            "Must be evenly divisible in the y axis"
        
        self.adj_bins = self.find_adjasent_bins() # create adjasent bins list
        
        
    def bin_vectors(self, vecs):
        """
        Bin vectors with cordinates (x, y, . . .) into the grid
        """
        x_bins = (vecs[:, 0] / self.bin_dx).floor()
        y_bins = (vecs[:, 1] / self.bin_dy).floor()
        output = y_bins * self.Nx + x_bins
        return output.long()
    
    
    def find_adjasent_bins(self):
        """
        Create a list of which bins are adjasent to which bins
        """
        adj_bins = [] 
        for i in range(self.Nk):
            # find the cordinates of each bin in the bin grid
            x, y = self.index_to_cords(i)
            # find the neightbours of that cordinate
            cordinates = self.neighbours(x, y, self.Nx, self.Ny)
            # convert the cordinates back into an index
            indexs = [ self.cords_to_index(x_, y_) for x_,y_ in cordinates ]
            # store the indexs in tensor form
            adj_bins.append(torch.tensor(indexs))
        return adj_bins
    
    
    def neighbours(self, x, y, x_max, y_max, r=1):
        """
        Find the neighbours in radius r to the x and y cordinate
        (this includes itself as a neighbour)
        """
        return [(x_, y_) for x_ in range(x-r, x+r+1)
                         for y_ in range(y-r, y+r+1)
                         if (#(x != x_ or y != y_) and   # not the center
                             (0 <= x_ < x_max) and # not outside x range
                             (0 <= y_ < y_max))]   # not outside y range
    
    
    def index_to_cords (self, i):
        return i%self.Nx, int(i/self.Nx)
    
    def cords_to_index(self, x, y):
        return y * self.Nx + x
       
        
        
class distance_metrics():
    """
    This holds all the different distance metric choices. Each function
    find the distance between every vector pair in two rank two tensors
    """
    
    def __init__(self, choice, **kwargs):
        
        if choice == 'normal':
            self.distance = self.distance_normal
            
        elif choice == 'default':
            self.distance = self.distance_scaled
            if 'factor' in kwargs.keys():
                self.factor = kwargs['factor']
            else:
                self.factor = 1
            
        elif choice == 'custom':
            if 'dist_func' in kwargs.keys():
                self.distance = kwargs['dist_func']
            else:
                raise(ValueError)
        
        else:
            raise(ValueError)
    
    
    def distance_normal(self, vecs, clusts):
        """
        Find the distance between every vector and every cluster
        vecs and clusts are 2 rank tensors with samples on the first dimension
        returns a rank 2 tensor with samples on the first dimension and distance
        to each cluster on the second
        """
        displacement = clusts - vecs[:, None]
        return displacement.norm(dim=2)

    
    def distance_scaled(self, vecs, clusts):
        """
        Find the distance between every vector and every cluster
        vecs and clusts are 2 rank tensors with samples on the first dimension
        returns a rank 2 tensor with samples on the first dimension and distance
        to each cluster on the second
        """
        col_clust, col_vec = clusts[:, 2:], vecs[:, 2:]
        pos_clust, pos_vec = clusts[:, :2], vecs[:, :2]
        
        col_dist = (col_clust - col_vec[:, None]).norm(dim=2)
        pos_dist = (pos_clust - pos_vec[:, None]).norm(dim=2)

        return col_dist + pos_dist * self.factor / np.sqrt(self.Np / self.Nk)
    
    

class SLIC(img_base, bin_base, distance_metrics):
    """
    Implements Kmeans clustering on an image in 5d color position space
    using localised bins on a regular grid to enforce locality.
    """
    
    def __init__(self, img, bin_grid, dist_metric='default', **kwargs):
        """
        Parameters
        ----------
        
        img : 3d numpy array
            Image to be segmented by kmeans,
            expected in shape [x_dim, y_dim, rgb] with all values in the
            interval [0, 1], not 255.
            
        bin_grid : tuple length 2
            The number of initial partitions in x and y respectivly,
            therefore the number of segments is the product of these. 
            Both must be a factor of the x, y dimensions for the whole
            image. This intial segmentation restrains the kmeans centers
            in space, forcing locality of segments and speeding up the
            algorithm.
            
        dist_metric : string, optional
            Choose the of calculating distance between two vectors:
            - 'normal' finds the normal distance without scaling either
                position or color space
            - 'default' finds the normal distance with scaling the position
                space by the bin widths. Can also pass the kwarg 'factor'
                with an additional scaling factor for the position space.
            - 'custom' pass a fucntion to be used in the kwarg 'dist_func'
        """
        
        # validate the kwargs
        valid_args = ['factor']
        for key in kwargs.keys():
            assert key in valid_args, "kwarg " + key + " was not recognised"
        
        # setup the image, bin_grid and distance metric base classes
        img_base.__init__(self, img, **kwargs)
        bin_base.__init__(self, bin_grid, **kwargs)
        distance_metrics.__init__(self, dist_metric, **kwargs)
        
        # which bin each vector belongs to
        self.vec_bins_tensor = self.bin_vectors(self.vectors)
        
        # which vectors are in each bin
        self.vec_bins_list = [(self.vec_bins_tensor==i).nonzero().squeeze()
                              for i in range(self.Nk)]
        
        # which cluster each vector belongs to
        self.cluster_tensor = self.vec_bins_tensor.clone()
        
        # which vectors are in each cluster
        self.cluster_list = [(self.cluster_tensor==i).nonzero().squeeze()
                             for i in range(self.Nk)]
        
        # initialise the centroids
        self.centroids = torch.empty([self.Nk, 5], dtype=torch.float)
        self.update_centroids()
        
        # stores the distances between vectors and local centroids
        self.vc_dists = []
        
        
    def update_centroids(self):
        """
        Find the new center of each cluster as the mean of its constituent
        elemtens
        """
        for i in range(self.Nk):
            vecs_in_cluster = self.cluster_list[i]
            assert vecs_in_cluster.numel() > 0, 'no cluser should be empty'
            self.centroids[i] = self.vectors[vecs_in_cluster].mean(dim=0)
            
            
    def update_distances(self):
        """
        Find the ditance between every vector in a bin and every centroid
        in that or the neighbouring bins.
        """
        
        # bin the centroids
        cent_bins_tensor = self.bin_vectors(self.centroids)
        
        # reset the distances
        self.vc_dists = []
        
        # for every bin grid (same as number of centroids)
        for i in range(self.Nk):
            
            # relevant centroids and vectors
            centroids_to_search = self.adj_bins[i]
            vecs_in_bin = self.vec_bins_list[i]
        
            # find distance between vectors in this bin and local centroids
            vecs_tensors = self.vectors[vecs_in_bin]
            cents_tensors = self.centroids[centroids_to_search]
            dist = self.distance(vecs_tensors, cents_tensors) 
            self.vc_dists.append(dist)
            
            
    def update_clusters(self):
        """
        Find which vectors belong to which cluster by taking the mimunum of
        the distances.
        """
        
        # for every bin grid (same as number of centroids)
        for i in range(self.Nk):
            
            # relevant centroids and vectors
            vecs_in_bin = self.vec_bins_list[i]
            centroids_to_search = self.adj_bins[i]
            
            # find which centroids are the closest and update them
            min_indexs = torch.argmin(self.vc_dists[i], dim=1)
            min_clusters = centroids_to_search[min_indexs]
            self.cluster_tensor[vecs_in_bin] = min_clusters
        
        # re-allocate which vectors are in each cluster
        for i in range(self.Nk):
            self.cluster_list[i] = (self.cluster_tensor==i).nonzero().squeeze()     
    
    
    def iterate(self, n_iter):
        "loop for n_iter iterations with progress bar"
        
        # create the progress bar
        self.progress_bar = progress_bar(n_iter)
    
        for i in range(n_iter):
            self.update_distances()
            self.update_clusters()
            self.update_centroids()
            self.progress_bar(i)


    def plot(self, option='default', ax=None, path=None, figsize=[22,22]):
        """
        Plot a one fo the following options on an axis if specified:
            - 'default' image and the segement outlines
            - 'setup' image, bin edges and cluster centers
            - 'segments' the sedment mask
            - 'edges' just the segment outlines in a transparent manner
            - 'img' just the orinal image
            - 'centers' each kmean centroid
            - 'bin_edges' the bin mesh used
            - 'bins' both 'img' and 'bin_edges'
            - 'time' the iterations vs time if iterate was called
            
        If path is given the image will be saved on that path.
        
        If no axis is given one can give a figsize wanted.
        """
        
        # validate opiton
        valid_ops = ['default', 'setup', 'edges', 'img', 'centers', 'bins', 'time',
                     'bin_edges', 'segments', 'setup']
        assert option in valid_ops, "option not recoginsed"
        
        # create an axis if not given
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)
            
        if option == 'default':
            self.plot('img', ax)
            self.plot('edges', ax)
            ax.set(title='Image Segmentation (default)')
        
        elif option == 'setup':
            self.plot('img', ax)
            self.plot('bin_edges', ax)
            self.plot('centers', ax)
            ax.set(title='Image Segmentation (setup)')
        
        elif option == 'segments':
            mask = self.tensor_to_mask(self.cluster_tensor)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (segments)')
        
        elif option == 'edges':
            mask = self.tensor_to_mask(self.cluster_tensor)
            mask = self.outline(mask)
            mask = self.mask_aplha_values(mask)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (edges)')
            
        elif option == 'img':
            ax.imshow(self.img)
            ax.set(title='Image Segmentation (img)')
            
        elif option == 'centers':
            ax.plot(self.centroids[:, 0].cpu().numpy(),
                    self.centroids[:, 1].cpu().numpy(), 'm*', ms=20)
            ax.set(title='Image Segmentation (centers)')

        elif option == 'bin_edges':
            mask = self.tensor_to_mask(self.vec_bins_tensor)
            mask = self.outline(mask)
            mask = self.mask_aplha_values(mask)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (bin_edges)')
            
        elif option == 'bins':
            mask = self.tensor_to_mask(self.vec_bins_tensor)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (bins)')
            
        elif option == 'time':
            assert hasattr(self, 'progress_bar'), 'must call iterate to use this'
            self.progress_bar.plot_time(ax)
            ax.set(title='Image Segmentation (time)')
            
        # save the figure if wanted
        if path:
            plt.savefig(path)
            
            
            
if __name__ == '__main__':
    # run an example
    
    # setup
    set_seed(10)
    img = get_img("images/TX1_white_cropped.tif")
    obj = SLIC(img, [20,15])

    # plot the initial binning 
    obj.plot("setup")
    plt.gca().set(title='Initial Grid')
    
    # iterate
    obj.iterate(10)
    
    # plot the resulting segmentation
    obj.plot('default')
    plt.gca().set(title='Segmentation after 10 Iterations')
    
    # plot the time taken
    obj.plot('time', figsize=[10, 10])