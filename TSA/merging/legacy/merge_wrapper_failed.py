class merge_wrapper():
    
    def __init__(self, img, mask, **kwargs):
        
        # validate image and mask
        assert img.shape[:2] == mask.shape, \
            'image and mask must have same x,y dimensions'
        
        # store given parameters
        self.img = img
        self.orig_mask = mask
        
        # derived constants
        self.y_dim, self.x_dim = mask.shape
        self.segment_ids = np.unique(mask)
        self.N = self.segment_ids.size
        
        # create the segment objects
        self.segments = []
        for i in range(self.N):
            cords = np.array(np.where(mask==i)[::-1]).T # careful as where returns y,x format
            self.segments.append(segment(cords, i, self))

            
        # create the id directory to track which segments have been merged
        self.directory = dict([(n, n) for n in self.segment_ids])
        
        
#     def find_edge_mask(self, arr):
#         lapacian = np.array([[-1, -1, -1],
#                              [-1,  8, -1],
#                              [-1, -1, -1]])
#         conv = sig.convolve2d(lapacian, arr, "valid")
#         return (conv > 0).astype('float')
                
    
    def neighbours(self, x, y):
        "find the neighbours to the x and y pixel"
        return np.array([(x_, y_) for x_ in range(x-1, x+2)
                                  for y_ in range(y-1, y+2)
                         if ((x != x_ or y != y_) and   # not the center
                             (0 <= x_ < self.x_dim) and # not outside x range
                             (0 <= y_ < self.y_dim))])  # not outside y range

    
    def compare(self, seg_1, seg_2):
        """
        Compare these two segments, and merge them if they meet the criteria
        """
        ############################# tmp comparison

        
        avg_col_diff = np.linalg.norm(seg_1['avg_color'] - seg_2['avg_color'])
        std_col_diff = np.linalg.norm(seg_1['std_color'] - seg_2['std_color'])
        
        if avg_col_diff < 0.01 and std_col_diff < 0.01:
            self.to_merge.append((seg_1.id, seg_2.id))
            
            
    def merge(self):
        """
        Merge two segments by 'killing' the 2nd and adding its pixels to the 
        1st. Also use the directory to make the 2nds id now point to the 1sts.
        """
        
        # validate no cells are marked to merge twice
        
        
        for seg_1_id, seg_2_id in self.to_merge:
            
            seg_1_id = self.directory[seg_1_id]
            seg_2_id = self.directory[seg_2_id]
            
            # find the combined pixels
            combined_pixels = np.hstack([seg_1.pixels, seg_2.pixels])

            # recaculate all the properties for the new segement without
            # creating a new object (kwargs still there from initaliseation)
            seg_1.__init__(combined_pixels, seg_1.id, self)
            seg_2.pixels = [None]

            # update the directory
            for k, v in self.directory.items():
                if v == seg_2.id:
                    self.directory[k] = seg_1.id
                

    def scan(self):
        """
        
        """
        # store segments already scanned so each pair only checked once
        already_scanned = []
        
        for seg_id in self.segment_ids:
            seg_id = self.directory[seg_id] # actual id incase merged
            
            if seg_id not in already_scanned: # incase merged and already searched
                seg_1 = self.segments[act_id]

                for neigh_id in seg_1.neighbours:
                    neigh_id = self.directory[neigh_id] # actual id incase merged

                    if neigh_id not in already_scanned: # if not already checked
                        seg_2 = self.segments[neigh_id]
                        self.compare(seg_1, seg_2)

            already_scanned.append(seg_id)

        self.merge() # merge any segments marked to merge
             


class segment():
    
    def __init__(self, pixels, index, wrapper, **kwargs):
        "pixels is a 2d array of shape (n, 2)"
        
        # store the given params
        self.pixels = pixels
        self.rgb = wrapper.img[pixels]
        self.id = index
        self.wrapper = wrapper
        
        # find the range
        self.range = np.array([[self.pixels[:,0].min(), self.pixels[:,1].min()],
                               [self.pixels[:,0].max(), self.pixels[:,1].max()]])
        
        # find the edges and the neighbours
        self.edges = self.find_edges()
        self.edge_dict = self.identify_edges()
        self.neighbours = self.edge_dict.keys()
        
        # make the metrics dirctionary
        self.metrics = self.calculate_metrics()
        
        
    def find_edges(self):
        "to be improved"
        
        edges = set()
        
        # loop over all x's
        for x in np.arange(*self.range[:, 0]):
            pixs = self.pixels[self.pixels[:, 0] == x]
            y_min = self.pixels[:, 1].min()
            y_max = self.pixels[:, 1].max()
            edges.add((x, y_min))
            edges.add((x, y_max))
            
        # loop over all y's
        for y in np.arange(*self.range[:, 1]):
            pixs = self.pixels[self.pixels[:, 1] == y]
            x_min = self.pixels[:, 0].min()
            x_max = self.pixels[:, 0].max()
            edges.add((x_min, y))
            edges.add((x_max, y))
            
        # store the edges
        return np.array(list(edges))

        
    def identify_edges(self):
        """
        Find the min and max points on every coordinate line that passes
        through this grain, ensuring to not store suplicates.
        """
        
        edge_dict = {}
        
        for x,y in self.edges:
            adj = self.wrapper.neighbours(x, y)
            # indexs backwards as original image is in y,x format
            indexs = self.wrapper.orig_mask[adj[:, 1], adj[:, 0]]
            
            for i in np.unique(indexs):
                if i != self.id:
                    # if index found before append to it, else create it
                    edge_dict.setdefault(int(i),[]).append((x,y))
        
        return edge_dict

    
    def calculate_metrics(self):
        """
        Calculate the properties this segment which will beujsed
        in the merging citeria
        """
        
        mets = {}
        
        mets['size'] = self.pixels.shape[0]
        mets['perimeter'] = self.edges.shape[0]
        
        mets['avg_color'] = self.rgb.mean(axis=0)
        mets['std_color'] = self.rgb.std(axis=0)
        
        return mets
        			 