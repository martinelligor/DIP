"""
                    Nome: Igor Martinelli   
                    NUSP: 9006336
                    SCC5830 - Digital Image Processing
                    2020/1
                    Assignment 2: Image Enhancement and Filtering
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
    r = imageio.imread('./input_and_output_imgs/'+filename) 
    
    M = int(input())
    S = int(input())
    n = None
    if(M == 1):
        n = int(input())

    param_1 = float(input())
    param_2 = int(input()) if(M == 2) else float(input())

    return r, M, S, n, param_1, param_2

def gaussian_kernel(x, sigma):
    """
        This funcion applies the gaussian kernel function given a point x and a sigma.

        Parameters
        ----------
            x: float
                A parameter of the gaussian kernel.

            sigma: float
                The standard deviation of gaussian kernel.

        Returns
        -------
            The value of the gaussian kernel applied on x, with std equals sigma.
    """
    return (1/(2*np.pi*np.square(sigma)))*np.exp(-(np.square(x)/(2*np.square(sigma))))

def scaling(I):
    """
        This function realizes the normalization operation in an image to transform it's pixels into 0-255 range.

        Parameters
        ----------
            I: (numpy.array)
                The image before the normalization.

        Returns
        -------
            The normalized image between 0-255 range.
    """
    return (((I-I.min())*255.0)/(I.max()-I.min()))

def bilateral_filter(I, n, sigma_s, sigma_r):
    """
        This function is responsible to apply the biletarel filtering on a given image I.

        Parameters
        ----------
            I: (numpy.array)
                A matrix representing image I given as input.

            n: int
                The size of the filter used on the convolution operation.

            sigma_s and sigma_r: float
                The parameters used in gaussian kernel function.

        Returns
        -------
            f: (numpy.array)
                The result image after the operations in the input image I.
    """
    factor = int(n/2)
    img_shape = I.shape
    f = np.zeros(img_shape)
    # padding image to apply the filter.
    I = np.pad(I, pad_width=factor, mode='constant', constant_values=0)
    # calculating the spatial component.
    euclidean_matrix =[np.sqrt(np.square(i-factor)+np.square(j-factor)) for (i,j) in np.ndindex(n,n)]
    g_s = np.vectorize(gaussian_kernel)(euclidean_matrix, sigma_s).reshape(n,n)
    
    for(i, j) in np.ndindex(img_shape):
        # the window filter.
        win = np.flip(np.flip(I[i:i+n, j:j+n], 0), 1).astype(np.float)
        # getting the range gaussian component
        g_r = np.vectorize(gaussian_kernel)((win - I[i+factor, j+factor]), sigma_r).astype(np.float)
        # calculating de W_p.
        W_p = np.multiply(g_s, g_r)
        # getting the new value of the pixel p.
        f[i, j] = (np.sum(np.multiply(W_p, win))/np.sum(W_p)).astype(np.uint8)

    return f

def get_kernel(kernel_op):
    """
        This function is only responsible to return the choosed kernel to the method 2.

        Parameters
        ----------
            kernel_op: int
                The option selected as input.

        Returns
        -------
            (numpy.array)
                The right kernel given an option.
    """
    if(kernel_op == 1):
        return np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    else:
        return np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

def laplacian_filter(r, c, kernel_op):
    """
        This function is responsible to apply the laplacian filter in the image r given as input.

        Parameters
        ----------
            r: (numpy.array)
                An image given as input.

            c: int
                A constant used to multiply matrix after the operation of convolution.

            kernel_op: int
                The kernel that is choosed to realize the convolution operation.
    """
    n = 1
    I_f = r.copy().astype(np.float)
    kernel = np.flip(np.flip(get_kernel(kernel_op), 0), 1)
    I = np.pad(r, pad_width=n, mode='constant', constant_values=0).astype(np.float)

    for(i, j) in np.ndindex(r.shape):
        I_f[i, j] = np.sum(np.multiply(I[i:i+3, j:j+3], kernel))

    f = scaling(np.multiply(c, scaling(I_f)) + r)

    return f

def vignette_filter(r, sigma_row, sigma_col):
    """
        This function is responsible to apply the operation of vignetting on the image r given as input.

        Parameters
        ----------
            r: (numpy.array)
                The image r given as input.

            sigma_row and sigma_col: float
                The parameters of sigma to apply the gaussian kernel in the rows and columns.

        Returns
        -------
            The image after the vignetting operation.
    """
    # applying the operations in the columns.
    W_col = np.asmatrix(np.vectorize(gaussian_kernel)(np.arange(-int(r.shape[0]/2), int(r.shape[0]/2)+1, 1), sigma_row)[:r.shape[0]])
    # applying the operations in the rows.
    W_row = np.asmatrix(np.vectorize(gaussian_kernel)(np.arange(-int(r.shape[1]/2), int(r.shape[1]/2)+1, 1), sigma_col)[:r.shape[1]])

    # applying the operations to realized the vignetting operation.
    f = scaling(np.multiply(np.matmul(W_col.T, W_row), r))
    
    return f

def apply_image_transformation(r, M, S, n, param_1, param_2):
    """
        This function is responsible to redirect the operation in image r, given the method M, choosed in input.

        Returns
        -------
            r, f: (numpy.array)
                The original and the transformed image after the operations.
    """
    
    if(M == 1):
        f = bilateral_filter(r, n, param_1, param_2)
    elif(M == 2):
        f = laplacian_filter(r, param_1, param_2)
    elif(M == 3):
        f = vignette_filter(r, param_1, param_2)
    else:
        raise "Choose the right method in the input. (Possible methos are 1, 2 or 3)."

    # comparing images.
    RSE = compare_images(r, f)
    # saving transformed image.
    if(S):
        imageio.imwrite('output_img.png', r)
    # printing value of the comparison in the screen.
    print(np.round(RSE, 4))

    return r, f
    
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

if __name__ == "__main__":
    r, M, S, n, param_1, param_2 = read_input()
    apply_image_transformation(r, M, S, n, param_1, param_2)