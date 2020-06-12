"""
                    Nome: Igor Martinelli   
                    NUSP: 9006336
                    SCC5830 - Digital Image Processing
                    2020/1
                    Assignment 1: Intensity Transformations
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

            transformation: int
                The type of the transformation to be executed.

            c: int
                Possible parameter if the transformation 2 is choosed. 

            d: int
                Possible parameter if the transformation 2 is choosed. 

            W: int
                Possible parameter if the transformation 4 is choosed. 

            l: float
                Possible parameter if the transformation 4 is choosed. 

            save: int
                Flag indicating if the transformed image needs to be saved or not.

    """
    filename = str(input()).rstrip() 
    input_img = imageio.imread(filename) 
    
    c, d, W, l = np.repeat(None, 4)

    transformation = int(input())
    save = int(input())

    if(transformation == 2):
        c = int(input())
        d = int(input())
    elif(transformation == 4):
        W = int(input())
        l = float(input())
    
    return input_img, transformation, c, d, W, l, save

def inversion(img):
    """
        Function that executes the inversion transform of the image.

        Parameters
        ----------
            img: (numpy.ndarray):
                The reference image given as input.

        Returns
        -------
            transformed_img: (numpy.ndarray)
                The img given as parameter after the transformation.
    """
    transformed_img = (255 - img)

    return transformed_img

def contrast_modulation(img, c, d):
    """
        Function that executes the constrast modulation of the reference image.

        Parameters
        ----------
            img: (numpy.ndarray):
                The reference image given as input.

            c: int
                The minimum range of the new transformation of the reference image.

            d: int
                The maximum range of the new transformation of the reference image.

        Returns
        -------
            transformed_img: (numpy.ndarray)
                The img given as parameter after the transformation.
    """
    transformed_img = ((img - img.min()) * ((d-c)/(img.max()-img.min()))) + c

    return transformed_img

def logarithmic_function(img):
    """
        Function that executes the logarithm transform of the image.

        Parameters
        ----------
            img: (numpy.ndarray):
                The reference image given as input.

        Returns
        -------
            transformed_img: (numpy.ndarray)
                The img given as parameter after the transformation.
    """
    transformed_img = 255*(np.log2(1+img.astype(float))/np.log2(1+img.max().astype(float)))

    return transformed_img

def gamma_adjustment(img, W, l):
    """
        Function that executes the gamma transform of the image.

        Parameters
        ----------
            img: (numpy.ndarray):
                The reference image given as input.

            W: int
                One of the parameters of the gamma adjustment.

            l: float
                The other parameter of the gamma adjustment.

        Returns
        -------
            transformed_img: (numpy.ndarray)
                The img given as parameter after the transformation.
    """
    transformed_img = W*np.power(img, l)

    return transformed_img

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
    RSE = np.sqrt(np.sum(np.square(transformed_image.astype(float) - reference_image.astype(float))))

    return RSE

def apply_intensity_transformation():
    """
        Function that receives the parameters by the read_input function and realizes the transformations
        that are passed by parameter 'transformation'. After realize the image transform, the function 
        realizes the comparison between the reference image and the transformed image and prints the value.
        Also, if the parameter save is equal 1, the transformed image will be saved.
    """
    # acquiring the input values.
    image, transformation, c, d, W, l, save = read_input()

    # applying transformations.
    if(transformation == 1):
        transformed_image = inversion(image)
    elif(transformation == 2):
        transformed_image = contrast_modulation(image, c, d)
    elif(transformation == 3):
        transformed_image = logarithmic_function(image)
    elif(transformation == 4):
        transformed_image = gamma_adjustment(image, W, l)   

    # comparing images.
    RSE = compare_images(image, transformed_image)
    # saving transformed image.
    if(save):
        imageio.imwrite('output_img.png', transformed_image)
    # printing value of the comparison in the screen.
    print(np.round(RSE, 4))

if __name__ == "__main__":
    apply_intensity_transformation()