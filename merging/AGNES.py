import numpy as np
import matplotlib.pyplot as plt
from tools import progress_bar

class AGNES():
    
    def __init__(self, features, join_by='max'):
        "Cluster by AGNES"
        
        # logs which track merges and there distances
        self.dist_log = []
        self.merge_log = []
        
        # vectors
        self._vecs = features.copy()
        self._N = features.shape[0]
        self._group_count = self._N
        
        # find distances between every vector
        self._dists = np.full((self._N, self._N), np.inf)
        for _id in range(self._N):
            self._find_distance(_id)
                               
        # option of distance recaulculation
        if join_by == 'max':
            self._join = self._join_by_max
        elif join_by == 'min':
            self._join = self._join_by_min
        else:
            raise(ValueError)
    
        
    def _calc_dist(self, vec, vec_array):
            return np.sqrt(((vec - vec_array)**2).sum(axis=1))

        
    def _find_distance(self, _id):
        "Calculte the distance between this vector and all others"
        
        # find distance to every point before this id
        self._dists[_id, :_id] = self._calc_dist(self._vecs[_id],
                                                 self._vecs[:_id])
        
        # find the distance to every pstdoint after this id
        self._dists[_id+1:, _id] = self._calc_dist(self._vecs[_id],
                                                   self._vecs[_id+1:])
        return
        
        
    def _join_by_min(self, id1, id2):
        
        # find the minimum of the two distances
        dists_id1 = np.hstack([self._dists[id1, :id1], self._dists[id1:, id1]])
        dists_id2 = np.hstack([self._dists[id2, :id2], self._dists[id2:, id2]])
        dists_min = np.minimum(dists_id1, dists_id2)
        
        # make merging points inf distance apart
        dists_min[id1] = dists_min[id2] = np.inf
        
        # set the id1 distances to the min_distances
        self._dists[id1, :id1] = dists_min[:id1]
        self._dists[id1:, id1] = dists_min[id1:]
        
        # set the id2 distances to inf
        self._dists[id2, :id2] = self._dists[id2:, id2] = np.inf
    
    
    def _join_by_max(self, id1, id2):
        
        # find the minimum of the two distances
        dists_id1 = np.hstack([self._dists[id1, :id1], self._dists[id1:, id1]])
        dists_id2 = np.hstack([self._dists[id2, :id2], self._dists[id2:, id2]])
        dists_max = np.maximum(dists_id1, dists_id2)
        
        # set the id1 distances to the min_distances
        self._dists[id1, :id1] = dists_max[:id1]
        self._dists[id1:, id1] = dists_max[id1:]
        
        # set the id2 distances to inf
        self._dists[id2, :id2] = self._dists[id2:, id2] = np.inf
        
        
    def iterate(self):
        
        iterate = self._group_count-1
        
        bar = progress_bar(iterate) # verbose
        
        for i in range(iterate):
            
            if self._group_count == 1:
                print('\nFully merged graph\n')
                break
            
            id1, id2 = np.unravel_index(self._dists.argmin(),
                                        self._dists.shape)
            
            self.dist_log.append(self._dists[id1, id2])
            self.merge_log.append((id1, id2))
            self._group_count -= 1
            
            self._join(id1, id2)
             
            bar(i) # verbose
        
        print('\n')
            
            
    def cluster_by_derivative(self, std=3., plot=True):
        
        # find the second deriative cutoff
        y = np.array(self.dist_log)
        dy2 = y[:-2] - 2*y[1:-1] + y[2:] # central
        cutoff = np.std(dy2) * std
        
        print('Clustering up to 2nd derivative', cutoff)
        index = np.argmax(dy2 > cutoff) + 1
        # argmax gives the first instance,
        # +1 because dy2 not calculated for first merger 
        groupings = self.cluster_by_index(index, plot)
        
        return groupings

    
    def cluster_by_index(self, index, plot=True):
        "Keep merges up to index"
        
        # this directory tracks what group a vector belongs to
        directory = dict([(n,n) for n in range(self._N)])
        # { vector id : group id }
    
        n_clusters = self._N - index
        print('Clustering into', n_clusters, 'segments')
    
        # for every merge up to the requiered index
        for i in range(index):
            new_group, old_group = self.merge_log[i]
    
            # update every vector that had this group
            for vec_id, group in directory.items():
                    if group == old_group:
                        directory[vec_id] = new_group
                        
        # rebase the group indexs to start from 1 and be consecutive 
        groupings = np.unique(list(directory.values()), return_inverse=True)[1]
        
        if plot:
            fig, ax = plt.subplots(figsize=[12, 8])
            col = ax.scatter(*self._vecs[:, :2].T, c=groupings)
            ax.set(title = str(groupings.max()) +
                   ' Clusters up to Index ' + str(index))
            plt.colorbar(col)
            
        return groupings
    
    
    def cluster_by_distance(self, cutoff_dist=1., plot=True):
        
        print('Clustering up to distance', cutoff_dist)
        index = np.array(self.dist_log < cutoff_dist).sum()
        groupings = self.cluster_by_index(index, plot)
        
        return groupings

    
    def cluster_distance_plot(self, option, ax=None, last_few=True):

        if not ax:
            fig, ax = plt.subplots(figsize=[12, 10])
            
        n = 25 if last_few else self._N
        y = np.array(self.dist_log)
        
        if option == 'dists':
            x = np.arange(0, self._N-1)
            ax.plot(x[-n:], y[-n:], '-o')
            ax.set(title='Join Distances')
            
        elif option == '1st':
            x = np.arange(1, self._N-1)
            dy = y[1:] - y[:-1] # forward
            ax.plot(x[-n:], dy[-n:], '-o')
            ax.set(title='First Diff')
            
        elif option == '2nd':
            x = np.arange(1, self._N-2)
            dy2 = y[:-2] - 2*y[1:-1] + y[2:] # central
            ax.plot(x[-n:], dy2[-n:], '-o')
            ax.set(title='Second Diff')
            
        # composit option
        elif option == 'all':
            plt.close()
            fig, axs = plt.subplots(2,2, figsize=[15,15])
            self.cluster_distance_plot('dists', ax=axs[0,0])
            self.cluster_distance_plot('dists', ax=axs[0,1], last_few=False)
            self.cluster_distance_plot('1st', ax=axs[1,0])
            self.cluster_distance_plot('2nd', ax=axs[1,1])
            
        else:
            plt.close()
            print('opt not known')
            
            
if __name__ == '__main__':
    
    # generate dummy 2d data
    np.random.seed(10)
    features = np.random.normal(size=[500, 2])
    features[50:, :] += 5
    features[100:, :] += 5
    features[150:, 0] += 5
    features[250:300, 1] += 10
    features[350:400, 0] += 10
    features[-1, -1] += 20
    
    plt.figure(figsize=[5, 5])
    plt.scatter(*features.T)
    plt.title("dummy data")

    obj = basic_heirachical_clustering(features)
    obj.iterate()
    obj.cluster_distance_plot('all')
    obj.cluster_by_derivative(std=3., plot=True)
    