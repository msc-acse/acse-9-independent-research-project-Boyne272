import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

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
        
        # setup
        seg_ids = np.unique(self.work_mask)
        segments = {}
        
        # verbose
        bar = progress_bar(len(seg_ids))
        print('Initalising segments')
        
        for n, i in enumerate(seg_ids):
            
            bar(n)
            
            # where the mask array has this id
            bool_arr = self.work_mask==i
            cords = np.array(np.where(bool_arr)[::-1]).T
            # [::-1] as np.where returns y,x format
            # T as we want in form (N, 2)
            
            # where the mask array is both an edge and has this id
            bool_arr = bool_arr * self.work_edges
            edges = np.array(np.where(bool_arr)[::-1]).T
            # [1:-1, 1:-1] so as to not consider outer edges
            
            segments[i] = (segment(cords, edges, i, self))
            
        print('\n')
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
        Compare these two segments, and return true if they meet the criteria
        to merge
        """
        m1, m2 = seg_1.metrics, seg_2.metrics
        vote_count = 0
        vote_thresh = 1
        
        # handel tiny segments
        # if the segment is surrunonded by the other segment
        if seg_1.neighbours == [seg_2.id] or seg_2.neighbours == [seg_1.id]:
             # if either segment is very small
            if m1['size'] < m2['size'] * 0.01 or m2['size'] < m1['size'] * 0.01:
                return True
        
        # voting system
        
        # if the diving line is very strait
#         if m1['straitness'][seg_2.id] < 50:
#             vote_count += 1
        
        # if these segments are similar in color
        avg_col_diff = np.linalg.norm(m1['avg_color'] - m2['avg_color'])
        std_col_diff = np.linalg.norm(m1['std_color'] - m2['std_color'])
        if avg_col_diff < 0.05:# and std_col_diff < 0.05:
            vote_count += 1
            
        # count the votes
        if vote_count >= vote_thresh: 
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
        counter = 1
        while True:
            
            print("Starting Iteration ", counter)
            
            self.scan() # find which segments need to be merged
            if not self.to_merge: # if to_merge is empty stop
                print('No segments to merge, terminating\n')
                break
            else:
                print("merging ", len(self.to_merge), " segments\n")
                self.merge() # merge all nessesary segments
            
            # recreate the segments to represent the new mask
            self.segments, self.seg_ids = self.make_segments()
            counter += 1
            
            
    def plot(self, option='default', ax=None, **kwargs):
        
        # validate input
        all_options = ['default', 'compare1', 'compare2', 'both', 'merged',
                       'original', 'img', 'orig_mask', 'orig_edges',
                       'merged_mask', 'merged_edges',
                       'seg_fill', 'seg_edge']
        assert option in all_options, "option " + option + " not recognised"
        
        # if no axis given create one
        if not ax:
            fig, ax = plt.subplots(figsize=[22, 22])
        
        # plot combined options
        if option == 'default' or option == 'compare1':
            self.plot('img', ax=ax)
            self.plot('merged_edges', ax=ax)
            
            # fill in every segment that no longer exists
            post_merge_indexs = list(self.directory.values())
            NLE = [i for i in np.unique(self.orig_mask) # no longer exists
                   if i not in post_merge_indexs
                   or post_merge_indexs.count(i) > 1]
            NLE_mask = np.isin(self.orig_mask, NLE)
            NLE_rgba = self.rgba(NLE_mask, opaqueness=.5)
            ax.imshow(NLE_rgba)
            ax.set(title='merging comparison 1')

        if option == 'compare2':
            self.plot('img', ax=ax)
            self.plot('orig_edges', ax=ax)
            
            # fill in every segment that was merged
            post_merge_indexs = list(self.directory.values())
            changed_indexs = list(set([i for i in post_merge_indexs if
                                       post_merge_indexs.count(i) > 1]))
            for index in changed_indexs:
                self.segments[index].plot(self.orig_mask.shape, ax=ax, 
                                          opt='fill')
            ax.set(title='merging comparison 2')
            
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
            
#         elif option == 'vertices':
#             self.plot('img', ax=ax)
#             self.plot('merged_edges', ax=ax)
#             for seg in self.segments.values():
#                 seg.plot(ax=ax, opt='vertices')
            
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
            self.segments[kwargs['seg']].plot(ax=ax, opt=option[4:])
            ax.set(title='segment ' + str(kwargs['seg']))
        
        
    def rgba(self, mask, color='r', opaqueness=1):
        "Take a 2d mask and return a 4d rgba mask for imshow overlaying"
        
        # create the transparent mask
        zeros = np.zeros_like(mask)
        rgba = np.dstack([zeros, zeros, zeros, mask*opaqueness])
        
        # set the correct color channel
        i = ['r', 'g', 'b'].index(color)
        rgba[:, :, i] = mask
            
        return rgba
    
    
    def outline(self, mask, opt='full'):
        """
        Take a 2d mask and use a laplacian convolution to find the segment 
        outlines for plotting. Option decides if all directions are to be
        included 'full' or just horizontal and vertical ones 'edge'
        """
        laplacian = np.ones([3, 3])
        
        if opt == 'full':
            laplacian[1, 1] = -8
            
        elif opt == 'edge':
            laplacian[1, 1] = -4
            laplacian[[0, 2, 0, 2], [0, 0, 2, 2]] = 0
            
        conv = sig.convolve2d(mask, laplacian, mode='valid')
        conv = np.pad(conv, 1, 'edge') # ignore edges
        not_edge = np.isclose(conv, 0)
        return 1. - not_edge

        
        
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
    
    
    def identify_edges(self):
        """
        For every edge use the wrapper mask to find what other segments
        share this edge and store them in a dictionary so that all pixels
        on a particular border can be indexed.
        """
        
        # dictionary for neighbour_id:[edge_cordinates]
        edge_dict = {}
#         vertex_dict = {}
        
        for x, y in self.edges:
            adj_cords = self.wrapper.neighbours(x, y)
            # indexs backwards as mask has y, x format
            adj_segs = self.wrapper.work_mask[adj_cords[:, 1], adj_cords[:, 0]]
            unique_segs = np.unique(adj_segs)
            
            # store cordinates that boarder each neighbour appropirately
            for i in unique_segs:
                if i != self.id:
                    
                    # if neighbour segment already in edge_dict append to it
                    # else create it as a new entry
                    edge_dict.setdefault(int(i),[]).append((x, y))
                    
#                     # if this is a vertex store it
#                     if len(unique_segs) > 2 or len(adj_cords) != 8:
#                         vertex_dict.setdefault(int(i), []).append((x, y))
                        
        # set every edge entry to an array
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
        
#         mets['straitness'] = dict([[i, self.find_straitness(i)]
#                                     for i in self.neighbours])
        
        return mets
    #############################
    
    
    def find_straitness(self, edge):
        """
        Calculate the avg straitness for the given edge. This is defined ....
        """
        def line_dist(cords, point_1, point_2):
            """
            Finds the shortest distance between cord and a strait line through
            point_1 and point_2
            """
            x0,y0, x1,y1, x2,y2 = *cords, *point_1, *point_2
            return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 -  y1*x1) / \
                np.linalg.norm(point_1 - point_2)
        
        e_cords = self.edge_dict[edge]
        n = len(e_cords)
        
#         point_1 = e_cords[:5, :].mean(axis=0)
#         point_2 = e_cords[-5:, :].mean(axis=0)

#         point_1 = e_cords[np.random.randint(0, n, 5)].mean(axis=0)
#         point_2 = e_cords[np.random.randint(0, n, 5)].mean(axis=0)

        point_1 = e_cords[np.argmin(e_cords[:, 0]), :]
        point_2 = e_cords[np.argmax(e_cords[:, 0]), :]
    
        # cancel if there are 5 or less than points on this edge
        if len(e_cords) < 20:
            return np.nan
        
        avg_dist = np.mean([line_dist(c, point_1, point_2) 
                            for c in e_cords])
        
        return avg_dist
        
    
    
    def plot(self, ax, opt='edge'):
        """
        Create a rgba matrix for this segments with the option to plot the:
            - 'edge' just give the outline of the image
            - 'fill' the entire segment, translucent to see the image beneath
            - 'vertices' the corders of each segement (REMOVED)
        """
        zeros = np.zeros(self.wrapper.orig_mask.shape)
        rgba = np.dstack([zeros, zeros, zeros, zeros])
        
        if opt == 'edge':
            for n_id, n_cords in self.edge_dict.items():
                ax.plot(n_cords[:,0], n_cords[:,1], 'x', ms=5, label=n_id)
            ax.legend()
                
        elif opt == 'fill':
            red, green, blue = np.random.rand(3) / 2
            rgba[self.cords[:,1], self.cords[:,0], 0] = red
            rgba[self.cords[:,1], self.cords[:,0], 1] = green
            rgba[self.cords[:,1], self.cords[:,0], 2] = blue
            rgba[self.cords[:,1], self.cords[:,0], 3] = 0.6
            ax.imshow(rgba)
            
#         elif opt == 'vertices':
#             for cords in self.vertex_dict.values():
#                 ax.plot(cords[:, 0], cords[:, 1], 'yo')
                
        
        
        
if __name__ == '__main__':
    
    # example
    mask = np.loadtxt('masks/example_mask.txt')
    img = get_img('images/TX1_white_cropped.tif')
    wrap = merge_wrapper(img, mask)

    wrap.iterate()

    wrap.plot()