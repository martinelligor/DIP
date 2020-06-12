"""
                    Nome: Igor Martinelli   
                    NUSP: 9006336
                    SCC5830 - Digital Image Processing
                    2020/1
                    Short Assignment 1: Image Restoration
"""

import imageio
import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift

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
            The normalized image between 0-maxg range.
    """
    return ((I-I.min())/(I.max()-I.min()))*maxg

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

def denoising(img, G):
    """Function that applies de denoising process on a image 'img' using the gaussian filter G.

    Parameters
    ----------
    img : (numpy.ndarray))
        An array representing the image given as input.
    G : (numpy.ndarray)
        An array representing the gaussian filter made with the parameters k and sigma.

    Returns
    -------
    (numpy.ndarray)
        The image after the denoising process.
    """
    # padding the filter so that it has the same size of the image.
    pad = (img.shape[0]//2)-G.shape[0]//2
    G_pad = np.pad(G, (pad, pad-1), "constant",  constant_values=0)

    # computing the Fourier transforms.
    R = np.multiply(fftn(img), fftn(G_pad))
    
    return np.real(fftshift(ifftn(R))), G_pad

def deblurring(G, gamma, H):
    """Function that implements the deblurring process of the noise image 'image_t', using gamma and G parameters.

    Parameters
    ----------
    G : (numy.ndarray)
        The transformed image after the denoising process.
    gamma : float
        Gamma parameter used in the constrained least squares method
    H : (numpy.ndarray)
        The same gaussian filter used in denoising process.

    Returns
    -------
    (numpy.ndarray)
        The image after the deblurring process.
    """
    maxd = G.max()
    # laplacian filter for the process.
    P = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
    # padding the filter G for the process.
    pad = (G.shape[0]//2)-P.shape[0]//2
    P = fftn(np.pad(P, (pad, pad-1), "constant",  constant_values=0))
    # computing the CLS method.
    F = np.multiply(np.divide(np.conj(H), np.square(np.abs(H))+(gamma*np.square(np.abs(P)))), fftn(G))

    return np.real(fftshift(ifftn(F))), maxd

def denoising_deblurring():
    """Function that applies the denoising and the deblurring process on a image given as input.
    """
    # getting parameters for the process.
    img, k, sigma, gamma = read_input()
    # constructing gaussian filter.
    G = gaussian_filter(k, sigma)
    # realizing the denoising process and scaling image.
    t_image, G_pad = denoising(img, G)
    t_image = scaling(t_image, img.max())
    # realizing the deblurring process and scaling image.
    t_image, maxd = deblurring(t_image, gamma, fftn(G_pad))
    t_image = scaling(t_image, maxd)

    # showing the output after the process.
    print('{:.1f}'.format(np.std(t_image[:])))

if __name__ == "__main__":
    denoising_deblurring()