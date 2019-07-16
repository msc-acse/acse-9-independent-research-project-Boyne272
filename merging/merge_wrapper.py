import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

class merge_wrapper():
    
    def __init__(self, img, mask, **kwargs):
        
        # validate image and mask
        assert img.shape[:2] == mask.shape, \
            'image and mask must have same x,y dimensions'
        
        # store mask arrays
        self.img = img
        self.orig_mask = mask
        self.orig_edges = self.outline(mask)
        self.work_mask = mask.copy()
        self.work_edges = self.orig_edges.copy()
        
        # derived constants
        self.y_dim, self.x_dim = mask.shape
        
        # create the segment objects
        self.segments, self.seg_ids = self.make_segments()
        
        # initalise working vairables
        self.to_merge = []
        self.directory = dict([(n, n) for n in self.seg_ids])
    
    
    def make_segments(self):
        "create the segment objects from the current working mask"
        seg_ids = np.unique(self.work_mask)
        segments = {}
        for i in seg_ids:
            
            # where the mask array has this id
            bool_arr = self.work_mask==i
            cords = np.array(np.where(bool_arr)[::-1]).T
            # [::-1] as np.where returns y,x format
            # T as we want in form (N, 2)
            
            # where the mask array is both an edge and has this id
            bool_arr = (self.work_mask==i) * (self.work_edges)
            edges = np.array(np.where(bool_arr)[::-1]).T
            # [1:-1, 1:-1] so as to not consider outer edges
            
            segments[i] = (segment(cords, edges, i, self))
            
        return segments, seg_ids
    
        
    def neighbours(self, x, y):
        """
        Find the neighbours to the x and y pixel. This method is stored on the
        wrapper as it knows the dimensions of the whole image.
        """
        return np.array([(x_, y_) for x_ in range(x-1, x+2)
                                  for y_ in range(y-1, y+2)
                         if ((x != x_ or y != y_) and   # not the center
                             (0 <= x_ < self.x_dim) and # not outside x range
                             (0 <= y_ < self.y_dim))])  # not outside y range
    
    
    ############################# 
    def compare(self, seg_1, seg_2):
        """
        Compare these two segments, and merge them if they meet the criteria
        """
        m1, m2 = seg_1.metrics, seg_2.metrics
        avg_col_diff = np.linalg.norm(m1['avg_color'] - m2['avg_color'])
        std_col_diff = np.linalg.norm(m1['std_color'] - m2['std_color'])
        
#         if avg_col_diff < 0.01 and std_col_diff < 0.01:
        if avg_col_diff < 0.05:
            return True
        else:
            return False
    #############################
    
    
    def scan(self):
        """
        Scan over the whole mask and compare every segment pair, being
        careful to only compare each pair once
        """
        
        already_scanned = [] # stores fully scanned segments
        
        for seg_1 in self.segments.values():
            for neigh_id in seg_1.neighbours: # for every segment pair
                if neigh_id not in already_scanned:
                    if self.compare(seg_1, self.segments[neigh_id]):
                        self.to_merge.append((seg_1.id, neigh_id))

            already_scanned.append(seg_1.id)
                
    
    def merge(self):
        """
        For every pair in the to_merge list set the working mask to replace
        the 2nd id with the 1st one. Caution is taken to keep track of
        who has merged so that merging with a segment which has already been
        merged is still possible.
        """
        
        for seg_id1, seg_id2 in self.to_merge:
            
            # select the up to date ids
            seg_id1 = self.directory[seg_id1]
            seg_id2 = self.directory[seg_id2]
            
            # update the working mask
            self.work_mask[self.work_mask == seg_id2] = seg_id1
            
            # update the directory which tracks merged segments incase one
            # trys to merge with a segment that no longer exists
            for key, val in self.directory.items():
                if val == seg_id2:
                    self.directory[key] = seg_id1
        
        # clear the merging list
        self.to_merge = []
                    
                    
    def iterate(self):
        "The loop to be carried out until the new mesh is made"
        while True:
            
            self.scan() # find which segments need to be merged
            if not self.to_merge: break # if to_merge is empty stop
            print("merging:\n" , self.to_merge)
            self.merge() # merge all nessesary segments
            
            # recreate the segments to represent the new mask
            self.segments, self.seg_ids = self.make_segments()
            
            
    def plot(self, option='default', ax=None, **kwargs):
        
        # validate input
        all_options = ['default', 'compare', 'both', 'merged', 'original',
                       'img', 'orig_mask', 'orig_edges', 'merged_mask',
                       'merged_edges', 'seg_fill', 'seg_edge']
        assert option in all_options, "option " + option + " not recognised"
        
        # if no axis given create one
        if not ax:
            fig, ax = plt.subplots(figsize=[22, 22])
        
        # plot recall options
        if option == 'default' or option == 'compare':
            self.plot('img', ax=ax)
            self.plot('orig_edges', ax=ax)
            
            # fill in every segment that has been merged
            post_indexs = list(self.directory.values())
            changed_indexs = list(set([i for i in post_indexs if
                                       post_indexs.count(i) > 1]))
            for index in changed_indexs:
                rgba = self.segments[index].rgba(self.orig_mask.shape,
                                                 opt='fill')
                ax.imshow(rgba)
            ax.set(title='merging comparison')
            
        elif option == 'both':
            self.plot('img', ax=ax)
            self.plot('orig_edges', ax=ax)
            self.plot('merged_edges', ax=ax)
            ax.set(title='pre and post merging edges')
            
        elif option == 'merged':
            self.plot('img', ax=ax)
            self.plot('merged_edges', ax=ax)
            ax.set(title='merged segments')
            
        elif option == 'original':
            self.plot('img', ax=ax)
            self.plot('orig_edges', ax=ax)
            ax.set(title='original segments')
            
        # plot base options
        elif option == 'img':
            ax.imshow(self.img)
            ax.set(title='original image')
        
        elif option == 'orig_mask':
            ax.imshow(self.orig_mask)
            ax.set(title='original mask')
            
        elif option == 'orig_edges':
            edges = self.outline(self.orig_mask)
            rgba = self.rgba(edges, color='r')
            ax.imshow(rgba)
            ax.set(title='original mask outline')
            
        elif option == 'merged_mask':
            ax.imshow(self.work_mask)
            ax.set(title='current merged mask')
            
        elif option == 'merged_edges':
            edges = self.outline(self.work_mask)
            rgba = self.rgba(edges, color='g')
            ax.imshow(rgba)
            ax.set(title='original mask outline')
        
        elif option == 'seg_edge' or option == 'seg_fill':
            assert 'seg' in kwargs.keys(),\
                "must specifiy which segment to plot"
            
            self.plot('img', ax=ax)
            rgba = self.segments[kwargs['seg']].rgba(self.orig_mask.shape,
                                                        opt=option[4:])
            ax.imshow(rgba)
            ax.set(title='segment ' + str(kwargs['seg']))
        
        
    def rgba(self, mask, color='r'):
        "Take a 2d mask and return a 4d rgba mask for imshow overlaying"
        
        zeros = np.zeros_like(mask)
        rgba = np.dstack([zeros, zeros, zeros, mask])
        
        i = ['r', 'g', 'b'].index(color)
        rgba[:, :, i] = mask
            
        return rgba
    
    
    def outline(self, mask):
        """
        Take a 2d mask and use a laplacian convolution to find the segment 
        outlines for plotting
        """
        ################################ what about a 4 laplacian?
        laplacian = np.ones([3, 3])
        laplacian[1, 1] = -8
        edges = sig.convolve2d(mask, laplacian, mode='valid')
        edges = np.pad(edges, 1, 'edge') # ignore edges
        
        return (edges > 0).astype(float)
        
		
		
class segment():
    
    def __init__(self, cords, edges, index, wrapper, **kwargs):
        
        # store the given params
        self.cords = cords
        self.edges = edges
        self.rgbs = wrapper.img[cords[:, 1], cords[:, 0]]
        self.id = index
        self.wrapper = wrapper
        
        # find the cordinate range
        self.x_range = np.array([self.cords[:,0].min(), 
                                 self.cords[:,0].max() + 1])
        self.y_range = np.array([self.cords[:,1].min(), 
                                 self.cords[:,1].max() + 1])

        # find which edges are with each segments
        self.edge_dict = self.identify_edges()
        self.neighbours = list(self.edge_dict.keys())
        
        # find this segments metrics
        self.metrics = self.calculate_metrics()
        
        
#     def find_edges(self):
#         """
#         Loop over every x and find the max and min y values, these are 
#         the top and bottom edges. Similarly for every y find min and max
#         x values which are the left and right edges. Caution to avoid
#         duplicate pixels.
#         """
        
#         # use a set to prevent adding same cordinate twice
#         edges = set()
        
#         # for every x, find the edge pixels in the y direction
#         for x in np.arange(*self.x_range):
#             cords_on_line = self.cords[self.cords[:, 0] == x]
            
            
            
#             if cords_on_line.size == 0:
#                 print(self.id, x)
#                 print(self.x_range)
#                 print(self.y_range)
#                 print(self.cords)
                
                
                
#             edges.add((x, cords_on_line[:, 1].min())) # bottom edge
#             edges.add((x, cords_on_line[:, 1].max())) # top edge
            
#         # for every y, find the edge pixels in the x direction
#         for y in np.arange(*self.y_range):
#             cords_on_line = self.cords[self.cords[:, 1] == y]
#             edges.add((cords_on_line[:, 0].min(), y)) # left edge
#             edges.add((cords_on_line[:, 0].max(), y)) # right edge
            
#         # convert set to array
#         return np.array(list(edges))
    
    
    def identify_edges(self):
        """
        For every edge use the wrapper mask to find what other segments
        share this edge and store them in a dictionary so that all pixels
        on a particular border can be indexed.
        """
        
        # dictionary for neighbour_id:[edge_cordinates]
        edge_dict = {}
        
        for x, y in self.edges:
            adj_cords = self.wrapper.neighbours(x, y)
            # indexs backwards as mask has y, x format
            adj_segs = self.wrapper.work_mask[adj_cords[:, 1], adj_cords[:, 0]]
            
            for i in np.unique(adj_segs):
                if i != self.id:
                    # if neighbour segment already in edge_dict append to it
                    # else create it as a new entry
                    edge_dict.setdefault(int(i),[]).append((x, y))
        
        # set every entry to an array
        for key, lst in edge_dict.items():
            edge_dict[key] = np.array(lst)
        
        return edge_dict
    

    #############################
    def calculate_metrics(self):
        """
        Calculate the properties of this segment which will be used
        in the merging decision process
        """
        
        mets = {}
        
        mets['size'] = self.cords.shape[0]
        mets['perimeter'] = self.edges.shape[0]
        
        mets['avg_color'] = self.rgbs.mean(axis=0)
        mets['std_color'] = self.rgbs.std(axis=0)
        
        return mets
    #############################
    
    
    def rgba(self, shape, opt='edge'):
        
        # create the array to plot
        zeros = np.zeros(shape)
        rgba = np.dstack([zeros, zeros, zeros, zeros])
        
        if opt == 'edge':
            for neigh_id, cords in self.edge_dict.items():
                red, green, blue = np.random.rand(3)
                rgba[cords[:,1], cords[:,0], 0] = red
                rgba[cords[:,1], cords[:,0], 1] = green
                rgba[cords[:,1], cords[:,0], 2] = blue
                rgba[cords[:,1], cords[:,0], 3] = 1
                
        elif opt == 'fill':
            red, green, blue = 0.5 + np.random.rand(3) / 2
            rgba[self.cords[:,1], self.cords[:,0], 0] = red
            rgba[self.cords[:,1], self.cords[:,0], 1] = green
            rgba[self.cords[:,1], self.cords[:,0], 2] = blue
            rgba[self.cords[:,1], self.cords[:,0], 3] = 0.7

        return rgba
        