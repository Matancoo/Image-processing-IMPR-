import copy
import math

import numpy as np
import scipy.signal
from imageio import imread, imwrite
from scipy import signal
from scipy.io import wavfile as wav
from scipy.ndimage.interpolation import map_coordinates
from skimage.color import rgb2gray


def fouierMatrix(N):
    """
    matrix conversion from spacial domain to frequency domain
    :param N: dimention of the matrix
    :return: a translation matrix of size N*N from the time domain to the frequency domain.
    Note the the rows of the matrix represent the fourier basis for each element.
    """
    row, col = np.meshgrid(np.arange(N), np.arange(N))
    matrix = np.exp((-2j * math.pi * row * col) / N)
    return matrix




def InVfouierMatrix(N):
    """
    matrix conversion from frequency domain to spacial domain
    :param N: dimention of the matrix
    :return: a translation matrix of size N*N from the time domain to the frequency domain.
    Note the the rows of the matrix represent the fourier basis for each element.
    """
    row, col = np.meshgrid(np.arange(N), np.arange(N))  
    matrix = np.exp((2j * math.pi * row * col) / N)
    return matrix


def DFT(signal):
    """
    function that transforms a signal into a complex fourier_signal
    :param signal: an array of dtype float64 with shape (N,)
    :return: complex fourier_signal of shape (N,). a signal that represents the magnitude.
    """
    N = len(signal)
    fourier_matrix = fouierMatrix(N)
    fourier_signal = np.dot(fourier_matrix, signal)
    # to rectify similar forms (N,) vs (N,1)
    fourier_signal = fourier_signal.reshape(signal.shape)

    return fourier_signal


# TODO: need also only to calculate fourier on N/2 elements 

def IDFT(fourier_signal):
    """
    function that transforms a fourier_signal into a complex signal
    :param fourier_signal: an array of dtype complex128 with shape (N,)
    :return: signal of type 'complex128' with shape (N,)
    """
    N = len(fourier_signal)
    inverse_fourier_matrix = InVfouierMatrix(N)
    signal = np.dot(inverse_fourier_matrix, fourier_signal)
    signal = signal.reshape(fourier_signal.shape)               # to rectify similar forms (N,) vs (N,1)

    return signal / N


def DFT2(image):
    """
    function that transforms an image into a complex fourier_image
    :param image:  an array of dtype float64 with shape (N,M) or (N,M,1) (greyscale)
    :return: complex fourier_image of inpute shape.

    """
    shape = copy.copy(image).shape

    rows = np.shape(image)[0]
    cols = np.shape(image)[1]

    image = image.reshape(rows, cols, )

    row_matrix = fouierMatrix(rows)
    col_matrix = fouierMatrix(cols)
    fourier_image = row_matrix.dot(image).dot(col_matrix)  
    fourier_image = fourier_image.reshape(shape)            # to rectify similar forms (N,M,) vs (N,M,1)


    return fourier_image



def IDFT2(fourier_image):
    """
    function that transforms a complex fourier image to image
    :param fourier_image: complex fourier_image of shape (N,N)
    :return: an array of dtype float64 with shape (N,N)
    # """
    shape = copy.copy(fourier_image).shape
    rows = fourier_image.shape[0]
    cols = fourier_image.shape[1]
    fourier_image = fourier_image.reshape(rows, cols, )
    image = IDFT(fourier_image).dot(InVfouierMatrix(cols))
    image = image.reshape(shape)
    return image/cols


def change_rate(filename, ratio):
    """
    function that changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header. A “fast forward” effect is created. Given a WAV file, this function saves the audio
    in a new file called change_rate.wav

    :param filename:  a string representing the path to a WAV file
    :param ratio:  a positive float64 representing the duration change. (assume 0.25 < ratio < 4)
    :return: change_rate.wav a 1D ndarray of dtype float64 representing the new sample points
    """
    sampleRate_orig, data = wav.read(filename)
    sampleRate_new = int(ratio * sampleRate_orig)
    wav.write("change_rate.wav", sampleRate_new, data)


def resize(data, ratio):
    """
     function that changes the number of samples by the given ratio.
     :param data: 1D ndarray of dtype float64 or complex128(*) representing the original sample points
     :param ratio:
     :return:  1D ndarray of the dtype float64 of data representing the new sample points
     """

    N = len(data)
    sample_dft = np.fft.fftshift(DFT(data))
    zeros = np.abs(N - (N / ratio))

    LEFT = int(np.floor(zeros / 2))
    RIGHT = int(np.ceil(zeros / 2))

    if ratio < 1:
        # pad with zeros each side of the frequencies (x-axis)
        resized_sample = np.pad(sample_dft, (LEFT, RIGHT))
    else:
        sample_dft = np.fft.fftshift(DFT(data))
        # trim each side of the frequencies (x-axis)
        sample_dft = sample_dft[LEFT:N - RIGHT]
        resized_sample = np.fft.ifftshift(sample_dft)
    return IDFT(resized_sample)


def change_samples(filename, ratio):
    """
    fast forward function that changes the duration of an audio file by reducing the number of samples
    using Fourier. This function does not change the sample rate of the given file

    :param filename:  a string representing the path to a WAV file,
    :param ratio: a positive float64 representing the duration change. (assume 0.25 < ratio < 4)
    :return: change_samples.wav is the SAME audio file but "fast forward"
     """
    rate, sample = wav.read(filename)
    new_sample = resize(filename, ratio)
    return wav.write('change_samples_wav', rate, new_sample)


# TODO why is this method better than fourier fastforward?
def resize_spectrogram(data, ratio):
    """
    function that speeds up a WAV file, without changing the pitch, using spectrogram scaling.
    This is done by computing the spectrogram, changing the number of spectrogram columns, and creating back
    the audio
    :param data: is a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: is a positive float64 representing the rate change of the WAV file (0.25 < ratio < 4)
    :return: the new sample points according to ratio with the same datatype as data.
    """
    data = stft(data)
    # TODO better understand axis trimming function
    resized_data = np.apply_along_axis(resize, 1, data, ratio)
    return istft(resized_data)


def resize_vocoder(data, ratio):
    """
    function that speedups a WAV file by phase vocoding its spectrogram.
    :param data: is a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: is a positive float64 representing the rate change of the WAV file (0.25 < ratio < 4)
    :return: n the given data rescaled according to ratio with the same datatype as data.
    """
    data_spectogram = stft(data)
    resized_data = phase_vocoder(data_spectogram, ratio)
    return istft(resized_data)


def conv_der(im):
    """
    function that computes the magnitude of image derivatives by  deriving the image in each
    direction separately (vertical and horizontal) and using simple convolution.
    :param im: grayscale images of type float64
    :return: magnitude of the derivative, with the same dtype and shape
    """

    kernel_dx = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
    kernel_dy = kernel_dx.T
    x_derivative = scipy.signal.convolve2d(im, kernel_dx, mode='same')
    # TODO: better understand differences of mode
    y_derivative = scipy.signal.convolve2d(im, kernel_dy, mode='same')
    magnitude = np.sqrt(np.abs(x_derivative) ** 2 + np.abs(y_derivative) ** 2)
    return magnitude


def fourier_der(im):
    """
    function that computes the magnitude of the image derivatives using Fourier transform
    :param im: float64 grayscale image
    :return: float64 grayscale image (magnitude of input image)
    """
    N = im.shape[0]
    M = im.shape[1]
    x_const = (2j * np.pi) / N
    y_const = (2j * np.pi) / M
    u = x_const * np.arange(-N // 2, N // 2)  # frequency x-axis
    v = y_const * np.arange(-M // 2, M // 2)  # frequency y-axis
    fourier_im = np.fft.fftshift(DFT2(im))
    x_derivative = IDFT2(v * fourier_im)
    # TODO: T.T is the only thing that works.
    y_derivative = IDFT2(u * fourier_im.T).T
    magnitude = np.sqrt(np.abs(x_derivative) ** 2 + np.abs(y_derivative) ** 2)
    return magnitude

# Project1 functions: #


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


# university provided functions: #


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(
        np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

x = np.array([1,1,1,1,1])
y = [2,2,2,2,2]
out = scipy.signal.convolve(x,x)
print(out)
z = np.pad(x,(3,6),mode="constant",constant_values = (4,6))
print(z)
np.convolve
DFT(x)
a = 5
