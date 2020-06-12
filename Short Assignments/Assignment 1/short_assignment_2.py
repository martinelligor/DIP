"""
                    Nome: Igor Martinelli   
                    NUSP: 9006336
                    SCC5830 - Digital Image Processing
                    2020/1
                    Short Assignment 1: Image Restoration
"""

import imageio
import numpy as np

def read_input():
    """
        This function is responsible to read the input to transform image.

        Returns
        -------
            input_img: (numpy.ndarray)
                A 2-D array representing the matrix of pixels of an img.

    """
    filename = str(input()).rstrip() 
    r = imageio.imread(filename) 
    
    k = int(input())
    sigma = float(input())
    gamma = float(input())

    return r, k, sigma, gamma

def scaling(I, maxg):
    """
        This function realizes the normalization operation in an image to transform it's pixels into 0-maxg range.

        Parameters
        ----------
            I: (numpy.array)
                The image before the normalization.

        Returns
        -------
            The normalized image between 0-255 range.
    """
    return (((I-I.min())*maxg)/(I.max()-I.min()))

def gaussian_filter(k=3, sigma=1.0):
    """Function that generates a gaussian filter given a k and sigma.

    Parameters
    ----------
    k : int, optional
        One of the parameters of gaussian filter, by default 3
    sigma : float, optional
        The standard deviation of the equation, by default 1.0

    Returns
    -------
    (numpy.ndarray)
        An array containig the gaussian filter generated.
    """

    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2)*(np.square(x) + np.square(y))/np.square(sigma))

    return filt/np.sum(filt)

def denoising(g, w):
    ''' A function to filter an image g with the filter w
    '''
    # padding the filter so that it has the same size of the image
    pad1 = (g.shape[0]//2)-w.shape[0]//2
    wp = np.pad(w, (pad1,pad1-1), "constant",  constant_values=0)

    # computing the Fourier transforms
    W = fftn(wp)
    G = fftn(g)
    R = np.multiply(W,G)
    
    r = np.real(fftshift(ifftn(R)))

def deblurring(t_image, gamma, G):
    maxd = t_image.max()
    P = fftn(np.array([[0,-1,0], [-1,4,-1], [0,-1,0]]))
    
    t_image = np.multiply(np.divide(t_image, np.add(np.square(np.abs(t_image)), gamma*np.square(np.abs(P)))), G)

    return t_image, maxd

def denoising_deblurring():
    img, k, sigma, gamma = read_input()

    G = gaussian_filter(k, sigma)

    t_image = denoising(t_image, G)
    t_image = scaling(t_image, img.max())

    t_image, maxd = deblurring(t_image, gamma, G)
    t_image = scaling(t_image, maxd)

    print(':.1f'.format(np.std(t_image)))

