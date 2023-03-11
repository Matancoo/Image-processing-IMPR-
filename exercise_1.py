import copy
import math

import numpy
import numpy as np

from skimage import data
from skimage.color import rgb2gray
from imageio import imread, imwrite
import matplotlib.pyplot as plt


def read_image(filename, representation):
    """

    :param filename: the filename of an image on disk (could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2)
    :return: image represented as matrix of type np.float64
    """
    impute_image = imread(filename)
    processed_image = impute_image.astype(np.float64)
    if representation == 1 and len(impute_image.shape) == 3:
        output_image = rgb2gray(processed_image)
    elif representation == 2:
        output_image = processed_image
    return output_image / 255  # normalized to the range {0,1}


def image_display(filename, representation):
    """

    :param filename:
    :param representation:
    :return:
    """
    image = read_image(filename, representation)
    plt.imshow(image, cmap='gray', vmin=0., vmax=1.)
    plt.show()



def rgb2yiq(imRGB):
    """

    :param imRGB: RGB image
    :return: image converted to YIQ
    """
    conversion_matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.275, -0.321],
        [0.212, -0.523, 0.311]
    ])
    return np.dot(imRGB, conversion_matrix.T)


def yiq2rgb(imYIQ):
    """

    :param imYIQ: YIQ image
    :return: image converted to RGB
    """
    conversion_matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.275, -0.321],
        [0.212, -0.523, 0.311]
    ])
    conversion_matrix = np.linalg.inv(conversion_matrix)
    return np.dot(imYIQ, conversion_matrix.T)


def histogram_equalize(im_orig):
    """
    :param im_orig: grayscale or RGB float64 image with values in [0, 1]
    :return: list [im_eq, hist_orig, hist_eq]
    im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1]
    hist_orig - is a 256 bin histogram of the original image (array with shape (256,)
    hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,)
    """

    im_grey = get_grey(im_orig)
    hist_orig = np.histogram(im_grey, 256)[0]
    hist_cumul = np.cumsum(hist_orig)
    # find the FIRST index -m- where the value in the cumulative histogram is not zero
    m = np.nonzero(hist_cumul)[0][0]
    
    # creating lookup table normalized and streched
    look_table = np.round(
        (
                (hist_cumul - hist_cumul[m]) / (hist_cumul[255] - hist_cumul[m])) * 255
    )
    look_table = look_table.astype(int)  # convert to int

    # creating  histogram for the equalized image
    hist_eq = np.zeros(hist_orig.shape, int)
    hist_eq[look_table] = hist_orig

    # Our image im_grey is a float image: in range [0, 1]. We need to convert it to int: [0, 255] so that we can
    # apply the lookup table.
    im_grey_new = (im_grey * 255).astype(int)
    # creating equalized image (dealing with RGB and Greyscale)
    im_eq = look_table[im_grey_new]  # change the original Y channel of the picture
    # The resulting image needs to be a "floating type" image with values in ([0, 1]), so need to convert back.
    im_eq = im_eq.astype(float) / 255
    # reassemble the Y with the YIQ and convert back to RGB
    if len(im_orig.shape) == 3:
        I = np.copy(rgb2yiq(im_orig)[:, :, 1])
        Q = np.copy(rgb2yiq(im_orig)[:, :, 2])
        im_eq = np.dstack([im_eq, I, Q])

        im_eq = yiq2rgb(im_eq)
        return [im_eq, hist_orig, hist_eq]

    return [im_eq, hist_orig, hist_eq]


def get_initial_boundaries(n_quant, hist_image):
    """
    function that calculates the intial boundaries of each segment in the picture's histogram so that in each segment there are
    equal number of pixels.
    :param n_quant:  is the number of intensities our output im_quant image should have
    :param hist_image: is a 256 bin histogram of the original image (array with shape (256,)
    :return: inital boundaries (np array)
    """
    boundaries = np.zeros(n_quant + 1, int)
    pixel_count = np.cumsum(hist_image)[255]
    hist_cumul = np.cumsum(hist_image)
    # evaluates how many pixels will be  each segment
    segment_num = pixel_count // n_quant
    for i in range(1, n_quant + 1):
        distance = abs(hist_cumul - segment_num * i)
        boundaries[i] = np.argmin(distance)
    return boundaries


def get_pixel_segment_avg(n_quant, hist_image, boundaries):
    """

    :param n_quant: is the number of intensities our output im_quant image should have
    :param hist_image: a 256 bin histogram of the original image (array with shape (256,)
    :param boundaries: boundaries of each segment in the picture's histogram (np array)
    :return:q1- python list of pixel averages corresponding to each segment
    """
    q1 = np.zeros(n_quant, int)
    boundaries = boundaries.astype(int)
    for i in range(n_quant):
        grey_levels = np.arange(boundaries[i], boundaries[i + 1])
        aa = np.sum(hist_image[boundaries[i]:boundaries[i + 1]] * grey_levels)
        total_seg_pixel = np.sum(hist_image[boundaries[i]:boundaries[i + 1]])
        q1[i] = (aa / total_seg_pixel)
    return q1


def get_updated_boundaries(n_quant, q1):
    """
    function that updates boundaries of each segment in the picture's histogram based on q1
    :param n_quant: is the number of intensities our output im_quant image should have
    :param q1: python list of pixel averages corresponding to each segment
    :return: boundaries of each segment in the picture's histogram (np array)
    """
    boundaries = np.zeros(n_quant + 1, int)
    for i in range(1, n_quant):
        boundaries[i] = np.ceil(np.average([q1[i - 1], q1[i]]))
    boundaries[n_quant] = 255
    return boundaries


def error_calculation(n_quant, q1, boundaries, hist_image):
    """

    :param n_quant: is the number of intensities our output im_quant image should have
    :param q1: python list of pixel averages corresponding to each segment
    :param boundaries: boundaries of each segment in the picture's histogram (np array)
    :param hist_image: a 256 bin histogram of the original image (array with shape (256,)
    :return:
    """
    total_error = 0
    boundaries = boundaries.astype(int)
    for i in range(n_quant):
        interval = np.arange(boundaries[i], boundaries[i + 1])
        distance = np.square(interval - q1[i])
        error = np.dot(distance, hist_image[boundaries[i]:boundaries[i + 1]])
        total_error += error
    return total_error


def get_grey(im_orig):
    """
    function that extracts from image (im_orig) the Y channel if image is RGB and Greyscale if image is Greyscale
    :param im_orig: the input grayscale or RGB image to be quantized (float64 image with values in [0, 1])
    :return: Y channel for RGB or Greyscale for Grayscale
    """
    if len(im_orig.shape) == 3:
        image = np.copy(rgb2yiq(im_orig)[:, :, 0])
    else:
        image = np.copy(im_orig)
    return image


def quantize(im_orig, n_quant, n_iter):
    """
    function that performs optimal quantization of a given grayscale or RGB image
    :param im_orig: the input grayscale or RGB image to be quantized (float64 image with values in [0, 1])
    :param n_quant: is the number of intensities our output im_quant  should have
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: a list [im_quant, error] where
        im_quant - is the quantized output image. (float64 image with values in [0, 1]).
        error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
                quantization procedure
    """
    
    image = get_grey(im_orig)
    hist_image = np.histogram(image, 256)[0]
    boundaries = get_initial_boundaries(n_quant, hist_image)
    boundaries = boundaries.astype(int)
    error = []
    
    for j in range(n_iter):
        q1 = get_pixel_segment_avg(n_quant, hist_image, boundaries)
        error.append(error_calculation(n_quant, q1, boundaries, hist_image))
        new_boundaries = get_updated_boundaries(n_quant, q1)

        if np.array_equal(boundaries, new_boundaries):
            break
        boundaries = new_boundaries

    im_quant = (np.copy(image) * 255).astype(int)
    boundaries = boundaries.astype(int)
    
    for i in range(n_quant):
        pixel_coordinates_to_change = (boundaries[i] < im_quant) & boundaries[i + 1] >= im_quant)  
        im_quant[pixel_coordinates_to_change] = q1[i]
    
    error = np.array(error)
    if len(im_orig.shape) == 3:
        I = rgb2yiq(im_orig)[:, :, 1]
        Q = rgb2yiq(im_orig)[:, :, 2]
        im_quant = im_quant.astype(float) / 255
        im_quant = np.dstack([im_quant, I, Q])
        im_quant = yiq2rgb(im_quant)
    else:
        im_quant = im_quant.astype(float) / 255

    return [im_quant, error]
