"""
                    Nome: Igor Martinelli   
                    NUSP: 9006336
                    SCC5830 - Digital Image Processing
                    2020/1
                    Short Assignment 3 - Mathematical Morphology for Color Image Processing
"""
import imageio
import numpy as np
from matplotlib.colors import rgb_to_hsv
from skimage.morphology import disk, opening, closing, dilation, erosion

def read_input():
    """This function is responsible to read the input to transform image.

    Returns
    -------
    input_img: (numpy.ndarray)
        A 2-D array representing the matrix of pixels of an img.
    """
    filename = str(input()).rstrip() 
    r = imageio.imread(filename) 
    
    k = int(input())
    option = int(input())

    return r, k, option

def scaling(I):
    """This function realizes the normalization operation in an image to transform it's pixels into 0-255 range.

    Parameters
    ----------
        I: (numpy.array)
            The image before the normalization.

    Returns
    -------
        The normalized image between 0-255 range.
    """
    return ((I-I.min())/(I.max()-I.min()))*255

def rgb_opening(img, k):
    """This function realizes the Opening operation to a RGB Image.

    Parameters
    ----------
    img : (numpy.ndarray)
        The RGB image given as input
    k : int
        The parameter to be used in the size of the structuring element.

    Returns
    -------
    (numpy.ndarray)
        The image after de opening operation.
    """
    # creating the structural element.
    d = disk(k)
    # creating a copy of the original image
    img_out = img.copy()
    # applying the opening operation in each channel of the image.
    for i in range(img.shape[2]):
        # applying opening operation
        img_out[:,:,i] = opening(img_out[:,:,i].astype(np.uint8), d)

    return img_out.astype(np.uint8)

def rgb_channels(img, k):
    """This function performs the operation on RGB channels of the image.

    Parameters
    ----------
    img : (numpy.ndarray)
        The RGB image given as input.
    k : int
        The parameter to be used in the size of the structuring element.

    Returns
    -------
    (numpy.ndarray)
        The new RGB image after the operations.
    """
    d = disk(k)
    # converting the image to HSV channel.
    img_out = rgb_to_hsv(img.copy())
    # normalizing image to 0-255 interval
    H = scaling(img_out[:,:,0])
    # getting the Red component of the new image.
    R = scaling(dilation(H.astype(np.uint8), d) - erosion(H.astype(np.uint8), d))
    # getting the Green component of the new image.
    G = opening(H.astype(np.uint8), d)
    # getting the Blue component of the new image.
    B = closing(H.astype(np.uint8), d)

    # returning the image as array.
    return np.dstack((R,G,B)).astype(np.uint8)


def mathematical_morphology():
    """This function realizes the mathematical morphology operations on the images used as test cases.
    """
    img, k, opt = read_input()

    if(opt==1):
        img_out = rgb_opening(img, k)
    elif(opt==2):
        img_out = rgb_channels(img, k)
    else:
        img_out = rgb_opening(img, 2*k)
        img_out = rgb_channels(img_out, k)

    compare_images(img, img_out)

def compare_images(reference_image, transformed_image):
    """
        This function realizes de comparison between the reference image and the transformed image.

        Parameters
        ----------
        reference_image: (numpy.ndarray)
            The image given as input.

        transformed_image: (numpy.ndarray)
            The transformed image after the applied transformations in reference_image.

        Returns
        -------
        RSE: float
            The value that represents the comparison between the two images.
    """
    n_elements = reference_image.shape[0]*reference_image.shape[1]
    RMSE = np.sqrt(np.sum(np.square(np.subtract(reference_image.astype(float), transformed_image.astype(float))))/n_elements)
    print('%.4f' % RMSE)

if __name__ == "__main__":
    mathematical_morphology()