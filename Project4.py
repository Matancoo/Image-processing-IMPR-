import random
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, convolve, map_coordinates
import shutil
from imageio import imwrite
import sol4_utils

# ___________________________VARIABLES______________________
FILER_SIZE = 3
FILTER = np.array([1, 0, -1]).reshape(3, 1)
N = 7
M = 7
RADIUS = 10
DESC_RAD = 3


def harris_corner_detector(im):
    """
    Detects harris corners.
    returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # compute matrix M consist; x,y derivatives and blurred x,y,xy
    im_x = convolve(im, FILTER)
    im_y = convolve(im, FILTER.T)
    x_squared = sol4_utils.blur_spatial(im_x ** 2, kernel_size=FILER_SIZE)
    y_squared = sol4_utils.blur_spatial(im_y ** 2, kernel_size=FILER_SIZE)
    xy = sol4_utils.blur_spatial(im_x * im_y, kernel_size=FILER_SIZE)
    det_M = x_squared * y_squared - xy * xy
    trace_M = x_squared + y_squared

    # computing eigenvalues of the matrix of each corrsponding pixels
    response_im = det_M - 0.04 * (trace_M ** 2)
    binary_im = non_maximum_suppression(response_im)
    feature_points = np.argwhere(binary_im)
    return np.flip(feature_points, axis=1)  # flip coordinates to get (col,row)


#
def create_patch(coord, dist):
    """
    creates a grid of coordinates from a center point
    :param coord: [x,y] coordinates of type list
    :param dist: distance from coord
    NOTE: the final grid will be of size (2*dist+1)*(2*dist+1)
    """
    x, y = coord[0], coord[1]
    x_axis = np.linspace(x - dist, x + dist, num=2 * dist + 1, dtype=int)
    y_axis = np.linspace(y - dist, y + dist, num=2 * dist + 1, dtype=int)
    X, Y = np.meshgrid(x_axis, y_axis)
    W = np.stack((X, Y), axis=2)
    return W


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    # """
    N = pos.shape[0]  # of coordinates in array pos
    K = 1 + 2 * desc_rad  # descriptor is matrix of size (K,K)
    descriptors = np.empty((N, K, K))
    for i in range(N):
        coord = pos[i]
        index_matrix = create_patch(coord, desc_rad).reshape((-1, 2)).T
        # The output is an interpolation of the value of the original array at the coordinates we specified
        descriptor = map_coordinates(im, index_matrix[[1, 0], :], order=1, prefilter=False)
        descriptor = descriptor.reshape((K, K)).T
        # Normalize descriptor to be invariant to intensity changes
        mean = np.mean(descriptor)
        norm = np.linalg.norm(descriptor - mean)
        if norm == 0:
            descriptor = np.zeros((K, K))
        else:
            descriptor = (descriptor - mean) / norm
        descriptors[i, :, :] = descriptor
    return descriptors


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """

    feature_points = spread_out_corners(pyr[0], m=7, n=7, radius=RADIUS)
    descriptors = sample_descriptor(pyr[2], feature_points / 4, desc_rad=DESC_RAD)  # TODO: maybe need conversion here
    return [feature_points, descriptors]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    N1, K, _ = desc1.shape
    N2, _, _ = desc2.shape
    s1 = desc1.reshape((N1, K * K))
    s2 = desc2.reshape((N2, K * K))
    s_N1xN2 = s1 @ s2.T

    best_col_indices_N1x2 = np.argpartition(s_N1xN2, -2)[:, -2:]
    best_row_indices_2xN2 = np.argpartition(s_N1xN2, -2, axis=0)[-2:, :]

    ### (j, k) -- (1) k is the 2nd or 1st best match of j of all ks
    ###           (2) j is the 2nd or 1st best match of k for all js
    ###           (3) S_{j,k} > thresh

    # For the 1st condition
    mask1 = np.zeros((N1, N2))
    # Fill the mask1 such that mask1[j, k] = 1 iff S_{j, k} satisfies condition (1) from the PDF
    for j in range(N1):
        for k in range(2):
            mask1[j, best_col_indices_N1x2[j, k]] = 1
    # For the 2nd condition
    mask2 = np.zeros((N1, N2))
    for j in range(N2):
        for k in range(2):
            mask2[best_row_indices_2xN2[k, j], j] = 1
    mask3 = s_N1xN2 > min_score
    # Which (i, j) are matches
    mask_matches = mask1 * mask2 * mask3

    return np.nonzero(mask_matches)


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    pos1 = np.insert(pos1, 2, 1, axis=1)  # adding (third column) containing ones
    pos = H12 @ pos1.T
    thrid_coord = pos[2, :]
    res = pos / thrid_coord
    return np.delete(res, 2, axis=0).T


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """

    N, _ = points1.shape
    P1 = np.copy(points1)
    P2 = np.copy(points2)
    max_inliers_count = 0
    final_index = None
    num_pts_needed = 1 if translation_only else 2
    for i in range(num_iter):
        # TODO: Need different way to get 2 *distinct* random numbers
        indexes = random.sample(range(N), num_pts_needed)
        H1 = estimate_rigid_transform(P1[indexes], P2[indexes], translation_only=translation_only)  # calc homography
        apprx_P2 = apply_homography(P1, H1)
        errors = np.linalg.norm(P2 - apprx_P2, axis=1)
        inliers_count = np.count_nonzero(errors < inlier_tol)  # count how many inliers we have
        if inliers_count > max_inliers_count:
            max_inliers_count = inliers_count
            final_index = np.argwhere(errors < inlier_tol).squeeze()
    # calculating final homography H
    H = estimate_rigid_transform(points1[final_index], points2[final_index], translation_only=translation_only)
    return [H, final_index]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    N, _ = points1.shape
    im = np.hstack((im1, im2))
    plt.imshow(im, cmap='gray')
    # plotting points
    shift_x = im1.shape[0]
    plt.scatter(x=points1[:, 0], y=points1[:, 1], c='r', s=3)
    plt.scatter(x=points2[:, 0] + shift_x, y=points2[:, 1], c='r', s=3)
    # plotting lines
    outliers = list(set(range(N)).difference(set(inliers)))  # is there a better way?

    for i in range(len(outliers)):
        plt.plot([points1[outliers[i]][0], points2[outliers[i]][0] + shift_x],
                 [points1[outliers[i]][1], points2[outliers[i]][1]],
                 mfc='r', c='b', lw=.4, ms=3,
                 marker='o')

    for i in range(len(inliers)):
        plt.plot([points1[inliers[i]][0], points2[inliers[i]][0] + shift_x],
                 [points1[inliers[i]][1], points2[inliers[i]][1]],
                 mfc='r', c='y', lw=.4, ms=3,
                 marker='.')  # inliers matches --> yellow line

    plt.show()




def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    N = len(H_succesive)
    H2m = [[0] * i for i in range(N+1)] #creating empty list
    forward_homo = np.identity(3,dtype='float64')
    # compute "left" homographies (i < m)
    for i in range(m - 1, -1, -1):
        forward_homo = np.dot(forward_homo,H_succesive[i])
        H2m[i] = (forward_homo)
    # compute "right" homographies (i > m)
    back_homo = np.identity(3,dtype='float64')
    for j in range(m, N): #TODO: I Think I got the indexing wrong---> check with avital
        inverse_homo = np.linalg.inv(H_succesive[j])
        back_homo = np.dot(back_homo,inverse_homo)
        H2m[j + 1] = back_homo
    # homography of frame m with itself (i == m)
    H2m[m] = np.identity(3,dtype='float64')
    # normalizing
    for i in range(len(H2m)):
        H2m[i] = H2m[i] / H2m[i][2, 2]
    return H2m



def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    up_left, up_right, down_right, down_left = [0, h - 1], [w - 1, h - 1], [0, 0], [w - 1, 0]
    corners = np.array([up_left, up_right, down_right, down_left])
    new_corners = apply_homography(corners, homography)
    min_x, min_y = min(new_corners[:, 0]), min(new_corners[:, 1])
    new_up_left = [min_x, min_y]
    max_x, max_y = max(new_corners[:, 0]), max(new_corners[:, 1])
    new_down_right = [max_x, max_y]
    return np.array([new_up_left, new_down_right]).astype(int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """

    width, height = image.shape[1], image.shape[0]
    up_left_point, down_right_point = compute_bounding_box(homography, width, height)
    x_min, x_max = up_left_point[0], down_right_point[
        0]
    y_max, y_min = down_right_point[1], up_left_point[1]
    # creating grid of inside bounding box
    x_axis = np.linspace(start=x_min, stop=x_max, num=np.abs(x_max - x_min) + 1,
                         dtype=int)  # TODO: could have useed arrange!! DOESNT WORK FOR NEGATIVES RUN2
    y_axis = np.linspace(start=y_min, stop=y_max, num=np.abs(y_max - y_min + 1), dtype=int)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    coords = np.stack((x_grid, y_grid), axis=-1).reshape((-1, 2))
    # get new coordinates by applying inverted homography
    inverse_coords = apply_homography(coords, np.linalg.inv(homography))
    # warp the channel to based on the new coordinates system
    warp_channel = map_coordinates(image, inverse_coords.T[[1, 0], :], order=1, prefilter=False)
    return warp_channel.reshape(y_grid.shape)  # TODO: WHY do I want to reshape it? only thing that works


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homo grahpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # # crop out areas not recorded from enough angles
        # # assert will fail if there is overlap in field of view between the left most image and the right most image
        # crop_left = int(self.bounding_boxes[0][1, 0])
        # crop_right = int(self.bounding_boxes[-1][0, 0])
        # assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        # print(crop_left, crop_right)
        # self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
        The bonus
        :param number_of_panoramas: how many different slices to take from each input image
        """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


