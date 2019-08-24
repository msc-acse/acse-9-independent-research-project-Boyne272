import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from tools import get_img

class Image_Filters:
    """
    A class simply to hold all image filters for more elegant code
    """
    
    def __init__(self):
        self.img_store = {}
        return
    
    # complex filters
    Sobel_0_3 = np.array([[-1, 0, 1],
                         [ -2, 0, 2],
                         [ -1, 0, 1]])
    Sobel_45_3 = np.array([[0, 1, 2],
                          [-1, 0, 1],
                          [-2,-1, 0]])
    Sobel_0_5 = np.array([[1, 2, 3, 2, 1],
                         [ 2, 3, 5, 3, 2],
                         [ 0, 0, 0, 0, 0],
                         [-2,-3,-5,-3,-2],
                         [-1,-2,-3,-2,-1]])
    Sobel_45_5 = np.array([[0, 2, 3, 2, 1],
                          [-2, 0, 3, 5, 2],
                          [-3,-3, 0, 3, 3],
                          [-2,-5,-3, 0, 2],
                          [-1,-2,-3,-2, 0]])
    Sobel_90_3 = Sobel_0_3.T
    Sobel_135_3 = Sobel_45_3.T
    Sobel_90_5 = Sobel_0_5.T
    Sobel_135_5 = Sobel_45_5.T
    Laws_L5 = np.array([1,4,6,4,1])
    Laws_E5 = np.array([-1,-2,0,2,1])
    Laws_S5 = np.array([-1,0,2,0,-1])
    Laws_W5 = np.array([-1,2,0,-2,1])
    Laws_R5 = np.array([1,-4,6,-4,1])

    
    def avg(self, size):
        return np.ones([size, size]) / (size*size)
    
    
    def laplacian_8(self, size):
        assert size%2==1, 'size must be odd'
        lap = np.ones([size, size])
        i = int(size/2)
        lap[i, i] = 1 - size*size
        return lap

    
    def gabor_fn(self, sigma, theta, Lambda, gamma=1., psi=0.):
        """
        Code from https://en.wikipedia.org/wiki/Gabor_filter#cite_note-8
        sigma: std of the gaussian
        theta: angle of the sinasoid in 2d
        Lambda: sinasoid wavelength
        psi: phase ofset of the sinasoid
        gamma: elpisive factor in the gaussian
        """
        sigma_x = sigma
        sigma_y = float(sigma) / gamma

        # Bounding box
        nstds = 3 # Number of standard deviation sigma
        xmax = max(abs(nstds * sigma_x * np.cos(theta)),
                   abs(nstds * sigma_y * np.sin(theta)))
        xmax = np.ceil(max(1, xmax))
        ymax = max(abs(nstds * sigma_x * np.sin(theta)),
                   abs(nstds * sigma_y * np.cos(theta)))
        ymax = np.ceil(max(1, ymax))
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1),
                             np.arange(xmin, xmax + 1))

        # Rotation 
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gauss = np.exp(-(x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2) / 2)
        trig = np.cos(2 * np.pi / Lambda * x_theta + psi)
        return gauss*trig
    
    
    def gabor_bank(self, n_theta, sigma, Lambda, gamma=1., psi=0.,
                   verbose=False):
    
        angs = np.linspace(0, 180, n_theta)
        filt_bank = [self.gabor_fn(sigma, np.deg2rad(ang), Lambda, gamma, psi)
                     for ang in angs]

        if verbose:
            x = 4
            y = int(n_theta/x) + 1
            fig, axs = plt.subplots(y, x, figsize=[16, 16])
            for filt, ax, ang in zip(filt_bank,
                                     axs.ravel()[:n_theta],
                                     angs):
                ax.imshow(filt, cmap='gray')
                ax.set(title=ang)
                ax.axis('off')
            
            # remove unused axis
            for ax in axs.ravel()[n_theta:]:
                fig.delaxes(ax)
                
        return filt_bank
    
    
    def apply_bank(self, filt_bank, grey_img, normed=True, verbose=False):
        output = np.zeros_like(grey_img)
        for filt in filt_bank:
            output += sig.convolve2d(grey_img, filt, mode='same')
               
        if normed:
            output -= output.min()
            output /= output.max()
        
        if verbose:
            fig, ax = plt.subplots(figsize=[22, 11])
            col = ax.imshow(output)
            plt.colorbar(col)
            
        return output
    
    
    def mask_outine(self, mask, size=3):
        """
        Take a 2d mask and use a laplacian convolution to find the segment 
        outlines for plotting. Option decides if all directions are to be
        included 'full' or just horizontal and vertical ones 'edge'
        """    
        conv = sig.convolve2d(mask, self.laplacian_8(size), mode='valid')
        conv = np.pad(conv, 1, 'edge') # ignore edges
        return 1. - np.isclose(conv, 0) # not zero grad (i.e. an edge)
    
    
    def sobel_edge_detection(self, grey_img, verbose=False):
        """
        Sobel edge detection convolution in 4 diretions returning a 3d array
        with angles (0, 45, 90, 135) on the final axis.
        """
        assert len(grey_img.shape) == 2, 'grey_img must be 2d'
        
        zeros = np.zeros_like(grey_img)
        edge_det = np.dstack([zeros, zeros, zeros, zeros])
        
        for i, filt in enumerate([self.Sobel_0_5, self.Sobel_45_5,
                                  self.Sobel_90_5, self.Sobel_135_5]):
            edge_det[:, :, i] = sig.convolve2d(grey_img, filt, mode='same')

        if verbose:
            for i in range(4):
                plt.figure(figsize=[22,22])
                plt.imshow(edge_det[:,:,i])
                plt.title('angle ' + str(i*45) + ' deg')
                
        return edge_det
    
         
    def rgba(self, mask, color='r', opaqueness=1):
        "Take a 2d mask and return a 4d rgba mask for imshow overlaying"
        
        # create the transparent mask
        zeros = np.zeros_like(mask)
        rgba = np.dstack([zeros, zeros, zeros, mask*opaqueness])
        
        # set the correct color channel
        i = ['r', 'g', 'b'].index(color)
        rgba[:, :, i] = mask
            
        return rgba
    
    
    def laws_texture_filters(self, size):
        
        filts = []
        arrs = [self.Laws_L5, self.Laws_E5, self.Laws_S5,
                self.Laws_W5, self.Laws_R5]
        arrs = [np.repeat(arr, size) for arr in arrs]
        
        for arr1 in arrs:
            for arr2 in arrs:
                filts.append(np.outer(arr1.T, arr2.T))
                
        return filts
    
    
    def laws_texture_features(self, grey_img, size, verbose=False):
        
        assert len(grey_img.shape) == 2, 'image must be grey'
        
        filts = self.laws_texture_filters(size)
        n_filts = len(filts)
        
        output = np.empty([*grey_img.shape, n_filts])
        
        for i, filt in enumerate(filts):
            output[:, :, i] = sig.convolve2d(grey_img, filt, mode='same')
            output[:, :, i] -= output[:, :, i].mean()
            output[:, :, i] -= output[:, :, i].std()
            
            if verbose:
                fig, ax = plt.subplots(figsize=[22,11])
                col = ax.imshow(output[:, :, i])
                plt.colorbar(col)
        
        return output            
        
        
        
if __name__ == '__main__':
    
    # gabor filter example
    IF_obj = Image_Filters()
    bank = IF_obj.gabor_bank(13, 5, 20, gamma=1., verbose=True)
    
    img = get_img('/content/images/TX1_white_cropped.tif')
    grey_img = img.mean(axis=2)
    
    IF_obj.apply_bank(bank, grey_img, verbose=True)

    # sobel_edge_detection example
    IF_obj = Image_Filters()
    
    img = get_img('/content/images/TX1_white_cropped.tif')
    grey_img = img.mean(axis=2)

    IF_obj.sobel_edge_detection(grey_img, verbose=True)