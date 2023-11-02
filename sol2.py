
import  numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import imageio
from skimage.color import rgb2gray

RGB = 2
def DFT(signal):
    """
    This function transforms a 1D discrete signal to its Fourier representation
    :param signal: is an array of dtype float64 with shape (N,) or (N,1)
    :return: complex Fourier signal
    """
    f_signal = signal
    if len(signal.shape) ==2:
        f_signal =signal.flatten()
    row = np.arange(0,signal.shape[0])
    col= np.array([row]).T
    w =np.exp((-2*np.pi*1j*row*col)/signal.shape[0])
    return_mat = np.dot(w, f_signal )
    if len(signal.shape) == 2:
        return_mat = return_mat.reshape(signal.shape[0],1)
    return  return_mat


def IDFT(fourier_signal):
    """
     This function transforms a  Fourier representation to its 1D discrete
     signal
    :param fourier_signal: is an array of type complex128
     with shape (N,) or (N,1)
    :return: complex signal
    """
    F_signal = fourier_signal
    if len(fourier_signal.shape) ==2:
        F_signal =fourier_signal.flatten()
    row = np.arange(0,fourier_signal.shape[0])
    col= np.array([row]).T
    w =np.exp((2*np.pi*1j*row*col)/fourier_signal.shape[0])
    return_mat = (np.dot(w,F_signal))*(1/fourier_signal.shape[0])
    if len(fourier_signal.shape) == 2:
        return_mat = return_mat.reshape(fourier_signal.shape[0],1)
    return  return_mat


def DFT2(image):
    """
    converts a 2D discrete signal to its Fourier representation
    :param image:  is a grayscale image of dtype float64
    :return: complex Fourier signal
    """
    new_image = image
    if len(image.shape) == 3:
        new_image = image.reshape(image.shape[0],image.shape[1])
    return_mat = dft2_iteration(image,new_image)

    if len(image.shape) == 3:
        return_mat = return_mat.reshape(image.shape[0],image.shape[1], 1)

    return return_mat

def dft2_iteration(image,new_image):
    """
    iterates over the rows and cols
    :param image: is a grayscale image of dtype float64
    :param new_image:  image with shape (image.shape[0],image.shape[1])
    :return: complex Fourier signal
    """
    return_mat = np.empty((image.shape[0], image.shape[1]),
                          dtype=np.complex128)
    for row in range(image.shape[0]):
        return_mat[row] = DFT(new_image[row])
    for col in range(image.shape[1]):
        return_mat[:, col] = DFT(return_mat[:, col])
    return  return_mat

def IDFT2(fourier_image):
    """
    convert a Fourier representation to its 2D discrete signal
    :param fourier_image: is a 2D array of dtype complex128

    :return:complex signal
    """
    new_image = fourier_image
    if len(fourier_image.shape) == 3:
        new_image = fourier_image.reshape(fourier_image.shape[0],
                                          fourier_image.shape[1])
    return_mat = idft2_iteration(fourier_image,new_image)

    if len(fourier_image.shape) == 3:
        return_mat = return_mat.reshape(fourier_image.shape[0],
                                        fourier_image.shape[1], 1)

    return return_mat 

def idft2_iteration(fourier_image, new_image):
    """
     iterates over the rows and cols
    :param fourier_image: is a 2D array of dtype complex128
    :param new_image:  image with new  shape
    :return: complex signal
    """
    return_mat = np.empty((fourier_image.shape[0], fourier_image.shape[1]),
                          dtype=np.complex128)
    for row in range(fourier_image.shape[0]):
        return_mat[row] = IDFT(new_image[row])
    for col in range(fourier_image.shape[1]):
        return_mat[:, col] = IDFT(return_mat[:, col])
    return return_mat

def change_rate(filename, ratio):
    """
    a function that changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header
    :param filename: is a string representing the path to a WAV file
    :param ratio: is a positive float64 representing the duration change.
    :return: none
    """
    sample_rate ,data = wavfile.read(filename)
    wavfile.write('change_rate.wav', int(sample_rate*ratio), data)


def change_samples(filename, ratio):
    """
    d function that changes the duration of an audio file by reducing the
    number of samples
    using Fourier. This function does not change the sample rate of the given file.
    :param filename:  is a string representing the path to a WAV file
    :param ratio: is a positive float64 representing the duration change.
    :return: 1D ndarray of dtype float64 representing the new sample points.
    """

    sample_rate, data = wavfile.read(filename)
    new_samples = resize(data,ratio)
    wavfile.write('change_samples.wav', int(sample_rate),
                           new_samples)

    return new_samples.astype('float64')

def resize(data, ratio):
    """
    changes the number of samples by the given ratio
    :param data: is a 1D ndarray of dtype float64 or complex128(*)
     representing the original sample points
    :param ratio:is a positive float64 representing the duration change.
    :return:  1D ndarray of the dtype of data representing
    the new sample points.
    """
    transfrom = DFT(data)
    shifted_array = np.fft.fftshift(transfrom)
    data_size = shifted_array.shape[0]
    new_data_size = np.floor(data_size/ratio)
    to_take = abs(data_size - new_data_size)
    if ratio ==1:
        return  data
    if ratio>1:
        new_samples =shifted_array [int(to_take / 2):
                                    int((data_size -np.ceil(to_take/2)))]

    if ratio<1:

        new_samples = np.pad(shifted_array, (int(np.floor((to_take / 2))),
                                             int((np.ceil(to_take/2)))),
                             'constant', constant_values=(0, 0))

    shifted_array = np.fft.ifftshift(new_samples)
    inverse_transform = IDFT(shifted_array)

    return inverse_transform.astype(data.dtype)



def resize_spectrogram(data, ratio):
    """
     function that speeds up a WAV file, without changing the pitch,
     using spectrogram scaling
    :param data:a is a 1D ndarray of dtype float64 representing the original
    sample points
    :param ratio:is a positive float64 representing
     the rate change of the WAV file
    :return: the new sample points according to ratio with the same datatype
     as data.
    """
    spectogram = stft(data)
    new_spectogram = np.apply_along_axis(resize,1,spectogram,ratio)
    return istft(new_spectogram).astype(data.dtype)



def resize_vocoder(data, ratio):
    """
    function that speedups a WAV file by phase vocoding its spectrogram
    :param data:a is a 1D ndarray of dtype float64 representing the original
     sample points
    :param ratio:is a positive float64 representing
    the rate change of the WAV file.
    :return:the given data rescaled according to ratio with the same datatype
     as data
    """
    vocoder = phase_vocoder(stft(data),ratio)
    return  istft(vocoder).astype(data.dtype)

def conv_der(im):
    """
    a function that computes the magnitude of image derivatives
    :param im:grayscale images of type float64
    :return: magnitude of the derivative
    """
    conv = np.array([0.5,0,-0.5])
    conv_x = conv.reshape((1,3))
    conv_y = np.array([conv]).T
    dx= signal.convolve2d(im,conv_x,mode = 'same')
    dy = signal.convolve2d(im,conv_y,mode ='same')
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude




def fourier_der(im):
    """
    function that computes the magnitude of the image derivatives
    using Fourier transform
    :param im:float64 grayscale image
    :return:magnitude of the derivative
    """
    transform_mat = DFT2(im)
    shifted_array = np.fft.fftshift(transform_mat)
    u= np.arange(-1*int(transform_mat.shape[1]/2),np.ceil((transform_mat.shape[1])/2))
    v=np.arange(-1*int(transform_mat.shape[0]/2),np.ceil((transform_mat.shape[0])/2))
    const_row = 2*np.pi*1j/transform_mat.shape[1]
    const_col =2*np.pi*1j/transform_mat.shape[0]
    mult_row = shifted_array*u
    mult_col =shifted_array*v.reshape(transform_mat.shape[0],1)
    array_row = np.fft.ifftshift(mult_row)
    array_col = np.fft.ifftshift(mult_col)
    dx= IDFT2(array_row)*const_row
    dy= IDFT2(array_col)*const_col
    magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
    return magnitude





def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    matrix_picture = imageio.imread(filename).astype('float64')
    if representation == RGB:
        return np.divide(matrix_picture, 255)

    return np.divide(rgb2gray(matrix_picture), 255)




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
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

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


