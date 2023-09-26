import numpy as np
from scipy import ndimage
import skimage.color as ski
from scipy.ndimage import convolve

from skimage.color import rgb2gray
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import os

UNIT_VECTOR = np.array([1])
BASE_VECTOR = np.array([1, 1])


# ___________Helper Functions_____________________
def gaussian_kernel(size):
    """
       :param size: -int- size of the given kernel (ODD NUMBER)
    :return: np array of shape (N,) of type float64. A normalized gaussian kernal
    """
    kernel = BASE_VECTOR
    for i in range(size - 2):
        kernel = np.convolve(kernel, BASE_VECTOR)
    kernel = kernel.reshape((1, size))
    return kernel / (2 ** (size - 1))  # normalize


def zero_pad(im):
    """
    function that pads every other element with zeros
    :param im: one channel image of shape (x,y) type float64
    :return: padded image
    """
    new_shape = (im.shape[0] * 2, im.shape[1] * 2)
    pad_image = np.zeros(new_shape)
    pad_image[::2, ::2] = im
    return pad_image


def expand(im, filter_vec):
    """
    function that expands a given image by a given size
    :param im: image greyscale of shape (x,y) of type float64
    :param filter_vec:  the kernel/filter used
    :return: new expanded image of shape (size)
    """
    expanded_im = zero_pad(im)
    row_kernel = 2 * filter_vec
    col_kernel = row_kernel.T
    new_im = convolve(expanded_im, row_kernel)
    new_im = convolve(new_im, col_kernel)
    return new_im


def reduce(im, size):
    """

    function that reduces a given image by a given size
    :param im: image greyscale of shape (x,y) of type float64
    :param size: the size of the kernel/filter used
    :return: new reduced image of shape (size)
    """

    row_kernel = gaussian_kernel(size)
    col_kernel = row_kernel.T
    im = convolve(im, row_kernel)
    im = convolve(im, col_kernel)
    return im[::2, ::2]


def build_gaussian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter (Assume >=2)
    :return:
    np.array --> (pyr, filter_vec)
    PYR: a standard python array with maximum length of max_levels, where each element of the array is agrayscale image
    FILTER_VEC: row vector of shape (1, filter_size) used for the pyramid construction.
    Built using a consequent 1D convolutions of [1,1] with itself in order to derive a row
    of the binomial coefficients which is a good approximation to the Gaussian profile. (NORMALIZED)
    """
    filter_vec = gaussian_kernel(filter_size)
    pyr = [im]
    pyr_level = np.copy(im)
    for i in range(1, max_levels):
        # make sure the last level of pyramid exceeds 16*16
        pyr_res = pyr_level.shape[0] * pyr_level.shape[1]
        if pyr_res <= 256:
            break
        pyr_level = reduce(pyr_level, filter_size)
        pyr.append(pyr_level)

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter (Assume >=2)

    :return: np.array --> (pyr, filter_vec)
    PYR: a standard python array with maximum length of max_levels, where each element of the array is agrayscale image
    FILTER_VEC: row vector of shape (1, filter_size) used for the pyramid construction.
    Built using a consequent 1D convolutions of [1,1] with itself in order to derive a row
    of the binomial coefficients which is a good approximation to the Gaussian profile. (NORMALIZED)
    """

    pyr = []
    pyr_gauss, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for level in range(len(pyr_gauss) - 1):
        lap_level = pyr_gauss[level] - expand(pyr_gauss[level + 1], filter_vec)
        pyr.append(lap_level)
    pyr.append(pyr_gauss[-1])
    return pyr, filter_vec


def expand_laplacian(lpyr, filter_vec):
    """
    function that expands each level in laplacian pyramid to the original image's shape.
    :param lpyr:laplacian pyramid (python array)
    :param filter_vec:row vector of shape (1, filter_size) used for the pyramid construction.
    :return: laplacian pyramid where each level is of shape of original image
    """
    for i in range(len(lpyr)):
        while lpyr[i].shape != lpyr[0].shape:
            lpyr[i] = expand(lpyr[i], filter_vec)
    return lpyr


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    function that implements the reconstruction of an image from its Laplacian Pyramid.
    :param lpyr: laplacian pyramid (python array)
    :param filter_vec: row vector of shape (1, filter_size) used for the pyramid construction.
    :param coeff: a python list. The list length is the same as the number of levels in the pyramid lpyr.
    :return:
    """
    lpyr = np.array(expand_laplacian(lpyr, filter_vec))
    coeff = np.array(coeff).reshape((-1, 1, 1))
    return np.sum(lpyr * coeff, axis=0)


def generate_filter_vec(filter_size):
    """
    gaussian filter approximation vector
    :param filter_size: size of gaussian filter
    :return: normalized filter vec
    """
    filter_vec = np.asarray([1])
    factor = 2 ** (filter_size - 1)
    for i in range(filter_size - 1):
        filter_vec = np.convolve(filter_vec, [1, 1])
    return (1 / factor) * filter_vec.reshape((1, filter_size))


def render_pyramid(pyr, levels):
    """
    function that stacks horizontally (after stretching the values to [0, 1]) the pyramid levels of the given pyramid
    :param pyr:  a Gaussian or Laplacian pyramid
    :param levels: # of levels to stack in the result ≤ max_levels
    :return: stack of pyramid levels (black picture)
    """
    image_height = pyr[0].shape[0]
    imgs = [pyr[0]]
    for level in range(1, levels):
        # stretch values of level to [0,1]
        pyr[level] = pyr[level] - pyr[level].min() / (pyr[level] - pyr[level].min()).max()
        new_img = np.zeros(shape=(image_height, pyr[level].shape[1]))
        new_img[:pyr[level].shape[0], :] = pyr[level]
        imgs.append(new_img)
    return np.concatenate(imgs, axis=1)


def display_pyramid(pyr, levels):
    """
       function that displays the stacked pyramid image.
       :param pyr: a Gaussian or Laplacian pyramid
       :param levels: # of levels to stack in the result ≤ max_levels
       :return: the displayed image
       """

    image = render_pyramid(pyr, levels)
    plt.imshow(image, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    function that blends two images using pyramid blending
    :param im1, im2: two input grayscale images to be blended
    :param mask: a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
    of im1 and im2 should appear in the resulting im_blend
    :param max_levels: parameter used when generating the Gaussian and Laplacian pyramids
    :param filter_size_im:  size of the Gaussian filter (an odd scalar that represents a squared filter) which
    defines the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: size of the Gaussian filter(an odd scalar that represents a squared filter) which
    defines the filter used in the construction of the Gaussian pyramid of mask.
    :return: blended image
    """
    lap1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask = np.array(mask, dtype='float64')
    mask_pyr = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    # building the new laplacian pyramid
    lap3 = []
    for i in range(len(lap1)):
        lap_level = mask_pyr[i] * lap1[i] + (1 - mask_pyr[i]) * lap2[i]
        lap3.append(lap_level)
    coeff = np.ones(len(lap1)).tolist()
    return laplacian_to_image(lap3, filter_vec, coeff)


def blending_example1():
    im1 = read_image('ronxx.jpeg', 2)
    im2 = read_image('catxx.jpeg', 2)
    mask = read_image('maskxx.jpeg', 1).astype(np.bool)
    im_blend0 = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask.astype(np.float64), 4, 3, 3)
    im_blend1 = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask.astype(np.float64), 4, 3, 3)
    im_blend2 = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask.astype(np.float64), 4, 3, 3)
    im_blend = np.stack((im_blend0, im_blend1, im_blend2), axis=2)
    im_blend[im_blend > 1] = 1
    im_blend[im_blend < 0] = 0

    # show
    plt.imshow(im1, vmax=1.0, vmin=0.0)
    plt.show()
    plt.imshow(im2, vmax=1.0, vmin=0.0)
    plt.show()
    plt.imshow(mask, cmap='gray')
    plt.show()
    plt.imshow((im_blend * 255).astype(np.uint8))
    plt.show()

    return im1, im2, mask, im_blend


def blending_example2():
    im1 = read_image('redback.png', 2)
    im2 = read_image('Ein.png', 2)
    mask = read_image('trueinmask.png', 1).astype(np.bool)
    im_blend0 = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask.astype(np.float64), 5, 5, 5)
    im_blend1 = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask.astype(np.float64), 5, 5, 5)
    im_blend2 = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask.astype(np.float64), 5, 5, 5)
    im_blend = np.stack((im_blend0, im_blend1, im_blend2), axis=2)
    im_blend[im_blend > 1] = 1
    im_blend[im_blend < 0] = 0

    # show
    plt.imshow(im1, vmax=1.0, vmin=0.0)
    plt.show()
    plt.imshow(im2, vmax=1.0, vmin=0.0)
    plt.show()
    plt.imshow(mask, cmap='gray')
    plt.show()
    plt.imshow((im_blend * 255).astype(np.uint8))
    plt.show()

    return im1, im2, mask, im_blend


# __________________PREVIOUS EX FUNCTIONS________________________
def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def read_image(filename, representation):
    """
    :param filename: the filename of an image on disk (could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2)
    :return: image represented as matrix of type np.float64
    """
    impute_image = imread(filename)
    processed_image = impute_image.astype(np.float64)
    if representation == 1:
        output_image = rgb2gray(processed_image)
    elif representation == 2:
        output_image = processed_image
    return output_image / 255  # normalized to the range {0,1}


# x = gaussian_kernel(3)
# y = np.outer(x,x)
# print(y)


def create_square_mask(im_dim):
    """
    Function that creates a binary mask of a back square
    """


mask = np.zeros((N1, N2))
# Fill the mask1 such that mask1[j, k] = 1 iff S_{j, k} satisfies condition (1) from the PDF
for j in range(N1):
    for k in range(2):
        mask1[j, best_col_indices_N1x2[j, k]] = 1

k = generate_filter(5)
print(k.shape)
