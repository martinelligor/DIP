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
    r = imageio.imread('../input_and_output_imgs/'+filename) 
    
    M = int(input())
    S = int(input())
    n = None
    if(M == 1):
        n = int(input())

    param_1 = float(input())
    param_2 = int(input()) if(M == 2) else float(input())

    return r, M, S, n, param_1, param_2

def gaussian_kernel(x, sigma):
    return (1/(2*np.pi*np.square(sigma)))*np.exp(-(np.square(x)/(2*np.square(sigma))))

def scaling(I):
    return ((I-I.min())*255)/I.max()

def bilateral_filter(I, n, sigma_s, sigma_r):
    factor = int(n/2)
    img_shape = I.shape
    f = np.zeros(img_shape)
    I = np.pad(I, pad_width=factor, mode='constant', constant_values=0)
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
    if(kernel_op == 1):
        return np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    else:
        return np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

def laplacian_filter(r, c, kernel_op):
    n = 1
    I_f = r.copy()
    kernel = get_kernel(kernel_op)
    I = np.pad(r, pad_width=n, mode='constant', constant_values=0).astype(np.float)

    for(i, j) in np.ndindex(r.shape):
        I_f[i, j] = np.sum(np.multiply(I[i:i+3, j:j+3], kernel))

    f = scaling(np.multiply(c, scaling(I_f)) + r)

    return f.astype(np.uint8)

def vignette_filter(r, sigma_row, sigma_col):
    W_col = np.asmatrix(np.vectorize(gaussian_kernel)(np.arange(-int(r.shape[0]/2), int(r.shape[0]/2)+1, 1), sigma_row)[:r.shape[0]])
    W_row = np.asmatrix(np.vectorize(gaussian_kernel)(np.arange(-int(r.shape[1]/2), int(r.shape[1]/2)+1, 1), sigma_col)[:r.shape[1]])

    return scaling(np.multiply(np.matmul(W_col.T, W_row), r)).astype(np.uint8)

def apply_image_transformation():
    r, M, S, n, param_1, param_2 = read_input()

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
    apply_image_transformation()