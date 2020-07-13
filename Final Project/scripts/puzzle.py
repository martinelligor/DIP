
"""
                    Nome: Igor Martinelli   
                    NUSP: 9006336
                    SCC5830 - Digital Image Processing
                    2020/1
                    Final Project: Remaining puzzle pieces counter
"""
import imageio
import numpy as np
import skimage.feature as sf		
from scripts.canny import Canny
import matplotlib.pyplot as plt

class PuzzleDetector():

    def __init__(self, filename=None, num_pixels=100, threshold=240):
        self.__NUM_PIECES = 0
        self.__IMAGE_MASK = None
        self.__GRAY_IMAGE = None
        self.__IMAGE_CANNY = None
        self.__FILENAME = filename
        self.__THRESHOLD = threshold
        self.__NUM_PIXELS = num_pixels
        self.__IMAGE = imageio.imread(filename)

    @property
    def NUM_PIECES(self):
        return self.__NUM_PIECES

    @NUM_PIECES.setter
    def NUM_PIECES(self, n):
        self.__NUM_PIECES = n

    @property
    def FILENAME(self):
        return self.__FILENAME

    @property
    def IMAGE_MASK(self):
        return self.__IMAGE_MASK

    @IMAGE_MASK.setter
    def IMAGE_MASK(self, mask):
        self.__IMAGE_MASK = mask

    @property
    def GRAY_IMAGE(self):
        return self.__GRAY_IMAGE

    @GRAY_IMAGE.setter
    def GRAY_IMAGE(self, g_img):
        self.__GRAY_IMAGE = g_img

    @property
    def IMAGE_CANNY(self):
        return self.__IMAGE_CANNY

    @IMAGE_CANNY.setter
    def IMAGE_CANNY(self, canny):
        self.__IMAGE_CANNY = canny

    @property
    def IMAGE(self):
        return self.__IMAGE

    @property
    def THRESHOLD(self):
        return self.__THRESHOLD

    @THRESHOLD.setter
    def THRESHOLD(self, threshold):
        self.__THRESHOLD = threshold

    @property
    def NUM_PIXELS(self):
        return self.__NUM_PIXELS

    @NUM_PIXELS.setter
    def NUM_PIXELS(self, num_pixels):
        self.__NUM_PIXELS = num_pixels

    def rgb2gray(self, img=None, ret=False):
        """This function is used to transform rgb image to grayscale image.

        Parameters
        ----------
        img_rgb : (numpy.ndarray)
            An image in RGB scale

        Returns
        -------
        (numpy.ndarray)
            The same image given as input in grayscale.
        """
        if img is None:
            if(ret):
                return np.floor(np.dot(self.IMAGE[...,:3], [0.299, 0.587, 0.114]))
            else:
                self.GRAY_IMAGE = np.floor(np.dot(self.IMAGE[...,:3], [0.299, 0.587, 0.114]))
        else:
            if(ret):
                return np.floor(np.dot(img[...,:3], [0.299, 0.587, 0.114]))
            else:
                self.GRAY_IMAGE = np.floor(np.dot(img[...,:3], [0.299, 0.587, 0.114]))

    def canny_edge_detection(self):
        self.IMAGE_CANNY = Canny(self.GRAY_IMAGE).canny()

    def image_segmentation(self):
        """This function creates a mask to segment an image.

        Parameters
        ----------
        image : (numpy.ndarray)
            The image to segment
        threshold : int
            A pixel value to be used as threshold.

        Returns
        -------
        (numpy.ndarray)
            A mask representing the pixels in the image that compound a piece.
        """
        # a mask to segment the image.
        mask = np.zeros(self.GRAY_IMAGE.shape)
        # if the pixel is above the threshold, his value in the mask is set to 255.
        mask[self.GRAY_IMAGE >= self.THRESHOLD] = 255
        
        self.IMAGE_MASK = mask

    def pieces_detection(self):
        """This function is responsible to realize the detection of how many pieces are in the image.

        Parameters
        ----------
        mask : (numpy.ndarray)
            A matrix containing the mask of the image, e.g., a representation 
            of the location of the pieces in the original image.

        Returns
        -------
        int
            The number of pieces in the image.
        """
        # The label for the piece #1.
        piece_label = 1
        # Random index to be used in flood-fill
        index = 200
        # Count of pixels that compose or not a piece.
        pixels = 0
        stack = []
        mask = self.IMAGE_MASK.copy()

        # For each pixel in image.
        for(i, j) in np.ndindex(mask.shape):
            # If the pixel is black, then verify his neighbors.
            if(mask[i, j] == 0):
                mask[i, j] = index
                stack.append([i, j])
                # while stack is not empty, check the neighborhood.
                while(len(stack) != 0):
                    # right side.
                    if(mask[i+1, j] == 0):
                        mask[i+1, j] = index
                        stack.append([i+1, j])
                        pixels += 1
                    # left side.
                    elif(mask[i-1, j] == 0):
                        mask[i-1, j] = index
                        stack.append([i-1, j])
                        pixels += 1
                    # underside.
                    elif (mask[i, j+1] == 0):
                        mask[i, j+1] = index
                        stack.append([i, j+1])
                        pixels += 1
                    # upside
                    elif (mask[i, j-1] == 0):
                        mask[i, j-1] = index
                        stack.append([i, j-1])
                        pixels += 1
                    # if none of the neighbors are 0, then pop one element and check his neighbors.
                    else:
                        (i, j) = stack.pop()
            # in the end of check step, if the number of pixels marked is greater then self.NUM_PIXELS, then, this
            # area is considered a piece and the pixels are labeled as a new piece.
            if(pixels > 100):
                mask[mask==index] = piece_label
                piece_label += 1
                pixels = 0

        self.IMAGE_MASK = mask.copy()
        # returns the number of pieces that the algorithm find in the image.
        self.NUM_PIECES = (piece_label-1)
        print("There are {} pieces of the puzzle in this image.".format(self.NUM_PIECES))

    def plot(self, mode):
        if(mode == 'img'):
            plt.figure(figsize=(8,12))
            plt.title("The image referent for the file {}.".format(self.FILENAME))
            plt.imshow(self.IMAGE)	

            plt.show()
        elif(mode == 'gray'):
            plt.figure(figsize=(8,12))
            plt.title("The image referent for the file {}.".format(self.FILENAME))
            plt.imshow(self.GRAY_IMAGE, cmap='gray')	

            plt.show()

        elif(mode == 'canny'):
            plt.figure(figsize=(8,12))
            plt.title("The image after Canny algorithm for the file {}.".format(self.FILENAME))
            plt.imshow(self.IMAGE_CANNY, cmap='gray')	

            plt.show()
        #para o modo mask, passando os arquivos, o mesmo gera a imagem que exibe as máscaras referentes aos casos de teste.
        elif(mode == 'mask'):
            plt.figure(figsize=(8,12))
            plt.title("The generated mask to the file {}.".format(self.FILENAME))
            plt.imshow(self.IMAGE_MASK, cmap='gray')	

            plt.show()
        #para o modo split, e, selecionando o índice do arquivo desejado, o mesmo exibe um gráfico exibindo a imagem com todas
        #as moedas e um gráfico para cada moeda identificada na imagem.
        elif(mode == 'split'):
            index = 1

            fig = plt.figure(figsize=(16,8))
            fig.add_subplot(4, 4, index)
            plt.title('The image to split')
            plt.imshow(self.IMAGE)

            for i in range(self.NUM_PIECES):
                plot = np.copy(self.IMAGE)
                plot[np.where(self.IMAGE_MASK != index)] = 255
                index += 1
                fig.add_subplot(4, 4, index)
                plt.imshow(plot)
                plt.title('Piece ' + str(index-1))
                
            plt.show()

    def quantization(self, img, b):
        """This function realizes the quantization of and image.

        Parameters
        ----------
        img : (numpy.ndarray)
            The image given as input to the class.
        b : int
            The parameter used to quantize image.

        Returns
        -------
        (numpy.ndarray)
            The image after the quantization.
        """
        return np.right_shift(np.uint8(img), (8-b))
    
    def color_histogram(self, img, C):
        """This function realizes the color histogram of the image.

        Parameters
        ----------
        img : (numpy.ndarray)
            The image given as input.
        C : int
            The number of bins in histogram.

        Returns
        -------
        (numpy.array)
            The histogram of the image.
        """
        # Number of bins: C = 2^b
        return (np.histogram(img, bins=C)[0])

    def haralick_texture_features(self, img):
        """The function that computes the Haralick texture features.

        Parameters
        ----------
        img : (numpy.ndarray)
            The image to compute the haralick texture features.

        Returns
        -------
        (numpy.array)
            An array containing the haralick features (energy, entropy, contrast, correlation and homogeneity)
        """
        # Calculating the GLCM.
        img = self.rgb2gray(img, ret=True).astype(np.uint8)
        G = sf.greycomatrix(img, [1], [1], )
        # Extracting Haralick descriptors.
        energy = sf.greycoprops(G, 'energy')[0][0]
        contrast = sf.greycoprops(G, 'contrast')[0][0]
        homogeneity = sf.greycoprops(G, 'homogeneity')[0][0]
        correlation = sf.greycoprops(G, 'correlation')[0][0]

        if(np.isnan(correlation)):
            correlation = 0
        # Calculating the entropy.
        G = np.asmatrix(G)
        entropy = -np.sum(G*np.log(G+1e-3))

        return np.array([energy, entropy, contrast, correlation, homogeneity])

    def gradient_orientation_descriptor(self, img):
        """Function that realizes the calculation of the orientation descriptor.

        Parameters
        ----------
        img : (numpy.ndarray)
            The image to compute the mathematics.

        Returns
        -------
        (numpy.array)
            The orientation descriptor.
        """
        img = self.rgb2gray(img, ret=True).astype(np.uint8).copy()
        # Sobel operators for x and y.
        wsx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        wsy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        
        gx = np.zeros(img.shape)
        gy = np.zeros(img.shape)
        # Calculating the gradients for x and y.
        m, n = img.shape	
        img = np.pad(img, 1, 'constant')
        for (i, j) in np.ndindex(m, n):
            gx[i][j] = np.sum(np.multiply(img[i:i+3, j:j+3], wsx))
            gy[i][j] = np.sum(np.multiply(img[i:i+3, j:j+3], wsy))

        M = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
        M = M/np.sum(M)

        # Calculating the angles and the histogram that compound the orientation descriptor.
        fi = (np.arctan2(gy, gx)*180.0/np.pi)+180.0
        dg = np.histogram(fi, bins=18, range=(0, 360), weights=M)[0]

        return dg

    def generate_descriptor(self, img, b):
        """This function realizes the creation of the descriptors of the image.

        Parameters
        ----------
        img : The image given as input to the class.
            The image given as input to the class.
        b : int
            The parameter used in color_histogram and gradient_orientation_descriptor functions.

        Returns
        -------
        (numpy.array)
            The array of features.
        """
        return (np.append(self.color_histogram(img, np.square(b)),
                np.append(self.haralick_texture_features(img),
                self.gradient_orientation_descriptor(img))))

    def generate_all_descriptors(self):
        """Thin function generates the descriptors of all the pieces in the image.

        Returns
        -------
        list
            A list containing the descriptors of each piece.
        """
        split_descriptors = []

        for index in range(self.NUM_PIECES):
            img = self.IMAGE.copy()
            img[np.where(self.IMAGE_MASK != index+1)] = 255
            img = self.quantization(img, 8)
            
            split_descriptors.append(self.generate_descriptor(img, 8))

        print('The descriptors of the {} images were generated'.format(self.NUM_PIECES))
        return split_descriptors

    def detect_equal_piece(self, piece_descriptor, descriptors, piece, C=np.square(8)):
        """This function receives a piece descriptor array and the descriptors of a bunch of pieces\
           and returns the piece that matchs better with the piece (class), given in the input too.

        Parameters
        ----------
        piece_descriptor : list
            A list containing the descriptor of a piece.
        descriptors : list
            A list containing the descriptors of a bunch of pieces.
        piece : PuzzleDetector
            An instance of PuzzleDetector class containing the parameters of the piece.
        C : int, optional
            The paremeter used in the auxiliar functions, by default np.square(8)
        """
        dc = piece.color_histogram(piece.IMAGE, C)
        dt = piece.haralick_texture_features(piece.IMAGE)
        dg = piece.gradient_orientation_descriptor(piece.IMAGE)
        # constructing the characteristhics array, composed by the three descriptors.
        do = np.append(dc, np.append(dt, dg))

        # weights of the descriptors.
        w_color = 6/10
        w_texture = 1/10
        w_gradient = 3/10
        # applying the weights in the array.
        w = np.zeros(len(do))
        w[0:np.size(dc)] = (1/np.size(dc))*w_color
        w[np.size(dc):np.size(dt)+np.size(dc)] = (1/np.size(dt))*w_texture
        w[np.size(dt)+np.size(dc):np.size(dc)+np.size(dt)+np.size(dg)] = (1/np.size(dg))*w_gradient
        # calculating the distance of the reference piece to each piece in the reference image.
        distances = []
        for descriptor in descriptors:
            distances.append(np.sqrt(np.sum(w*np.power((piece_descriptor-descriptor), 2))))
        # finding the piece in the reference image that matches with the reference piece.
        piece_num = np.argmin(distances)+1
        img = self.IMAGE.copy()
        img[np.where(self.IMAGE_MASK != piece_num)] = 255

        print('The most similar piece in the presented pieces is piece {}'.format(piece_num))

        fig = plt.figure(figsize=(16,8))

        fig.add_subplot(1,3,1)
        plt.imshow(self.IMAGE)
        plt.title('Reference image to search')

        fig.add_subplot(1,3,2)
        plt.imshow(piece.IMAGE)
        plt.title('The piece to compare.')
        
        fig.add_subplot(1,3,3)
        plt.imshow(img)
        plt.title('The more similar piece in the reference image.')

        plt.tight_layout()
        plt.show()