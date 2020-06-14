"""
                    Nome: Igor Martinelli   
                    NUSP: 9006336
                    SCC5830 - Digital Image Processing
                    2020/1
                    Final Project: Remaining puzzle pieces counter
"""
import numpy as np
from scipy.ndimage.filters import convolve

class Canny:
    """Class responsible to apply the canny edge detector algorithm in an image and the parameters given as input.
    """
    def __init__(self, img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, low_threshold=0.05, high_threshold=0.15):
        self.img = img
        self.sigma = sigma
        self.weak_pixel = weak_pixel
        self.kernel_size = kernel_size
        self.strong_pixel = strong_pixel
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def gaussian_kernel(self, size, sigma=1.0):
        """Function that generates a gaussian filter given a k and sigma. That step is used in Canny edge detection
           algorithm to smooth image in order to remove noise.

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
        arx = np.arange((-size // 2) + 1.0, (size // 2) + 1.0)
        x, y = np.meshgrid(arx, arx)
        filt = np.exp(-(1/2)*(np.square(x) + np.square(y))/np.square(sigma))

        return filt/np.sum(filt)
    
    def sobel_filters(self, img):
        """Function that generates sobel filters to be used in the canny edge detection process. This step intensifies
           the gradients of the image.

        Parameters
        ----------
        img : (numpy.ndarray)
            The 2-D array that represents the image

        Returns
        -------
        (numpy.ndarray)
            The image after the sobel operation.
        """
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float64)

        Ix = convolve(img, Kx)
        Iy = convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    

    def non_maximum_suppression(self, img, d):
        """Function that applies the non maximum suppression step in the canny edge detector

        Parameters
        ----------
        img : (numpy.ndarray)
            The image with the gradients intensified
        d : [type]
            [description]

        Returns
        -------
        (numpy.ndarray)
            The same image after non maximum supression step.
        """
        m, n = img.shape
        z = np.zeros((m, n), dtype=np.int64)
        angle = d*180./np.pi
        angle[angle < 0] += 180

        for i in range(1, m-1):
            for j in range(1, n-1):
                try:
                    q = 255
                    r = 255

                    #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        z[i,j] = img[i,j]
                    else:
                        z[i,j] = 0
                except:
                    pass
        return z

    def threshold(self, img):
        """This function applies a threshold in the image to provide a more accurate representation of real edges in an image.

        Parameters
        ----------
        img : (numpy.ndarray)
            The image after non maximum supression step.

        Returns
        -------
        (numpy.ndarray)
            The same image as input after the double threshold step.
        """
        high_threshold = img.max() * self.high_threshold
        low_threshold = high_threshold * self.low_threshold

        res = np.zeros(img.shape, dtype=np.int64)

        res[np.where(img >= high_threshold)] = np.int64(self.strong_pixel)
        res[np.where((img <= high_threshold) & (img >= low_threshold))] = np.int64(self.weak_pixel)

        return res

    def hysteresis(self, img):
        """This function is resposible to realize the edge tracking by hysteresis.

        Parameters
        ----------
        img : (numpy.ndarray)
            The image after the double-threshold step.

        Returns
        -------
        (numpy.ndarray)
            The same image given as input after the edge tracking step.
        """
        m, n = img.shape
        for i in range(1, m-1):
            for j in range(1, n-1):
                if (img[i,j] == self.weak_pixel):
                    try:
                        if ((img[i+1, j-1] == self.strong_pixel) or (img[i+1, j] == self.strong_pixel) or (img[i+1, j+1] == self.strong_pixel)
                            or (img[i, j-1] == self.strong_pixel) or (img[i, j+1] == self.strong_pixel)
                            or (img[i-1, j-1] == self.strong_pixel) or (img[i-1, j] == self.strong_pixel) or (img[i-1, j+1] == self.strong_pixel)):
                            img[i, j] = self.strong_pixel
                        else:
                            img[i, j] = 0
                    except:
                        pass

        return img
    
    def canny(self):
        """This function realizes the five steps in the image to acquire the edges of the image.

        Returns
        -------
        (numpy.ndarray)
            The detected edges of the image.
        """
        self.img_smoothed = convolve(self.img, self.gaussian_kernel(self.kernel_size, self.sigma))
        self.gradient, self.theta = self.sobel_filters(self.img_smoothed)
        self.non_maximum_suppression_img = self.non_maximum_suppression(self.gradient, self.theta)
        self.threshold_img = self.threshold(self.non_maximum_suppression_img)
        img_final = self.hysteresis(self.threshold_img)

        return img_final