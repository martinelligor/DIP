"""
                    Nome: Igor Martinelli   
                    NUSP: 9006336
                    SCC5830 - Digital Image Processing
                    2020/1
                    Short Assignment 1: Filtering in Fourier Domain
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
    
    T = float(input())

    return r, T


def DFT2D(f, inverse=False):
    """Function that apllies the discrete fourier transform.

    Parameters
    ----------
    f : (numpy.ndarray)
        The representation of the image as array.
    inverse : bool, optional
        If True, executes the inverse function of discrete fourier transform, by default False.

    Returns
    -------
    (numpy.ndarray )
        The image after apply the transformation.
    """

    F = np.zeros(f.shape, dtype=np.complex64)
    n, m = f.shape[0:2]

    x = np.arange(n).reshape(n,1)
    y = np.arange(m).reshape(1,m)

    for u in np.arange(n):
        for v in np.arange(m):
            if(inverse):
                F[u,v] = np.sum(np.multiply(f, np.exp(1j*2*np.pi*((((u*x)/n)+((v*y)/m))))))
            else:
                F[u,v] = np.sum(np.multiply(f, np.exp(-1j*2*np.pi*((((u*x)/n)+((v*y)/m))))))

    return F/np.sqrt(n*m)

def fourier_filtering():
    # getting inputs
    image, T = read_input()
    # aplying fast fourier 2-d transform on image.
    t_image = DFT2D(image)
    # getting the spectrum and computing the threshold.
    spectrum = np.abs(np.sqrt(np.square(np.real(t_image))+np.square(np.imag(t_image))))
    threshold = np.sort(spectrum.flatten())[-2]*T
    # aplying the threshold filter step.
    n_coefs = len(t_image[np.abs(t_image)<threshold].flatten())
    t_image[np.abs(t_image)<threshold] = 0
    # aplying the inverse fast fourier 2-d transform on image.
    t_image = np.abs(DFT2D(t_image, inverse=True))


    print('Threshold={:.4f}\nFiltered Coefficients={:d}\nOriginal Mean={:.2f}\nNew Mean={:.2f}'.format(threshold, n_coefs,
                                                                                                np.mean(image), np.mean(np.real(t_image))))

if __name__ == "__main__":
    fourier_filtering()
    
