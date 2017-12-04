#!/usr/bin/env python2

import numpy as np
import os
from scipy.signal import decimate

# This lets you set how time is reflected in the sampling
SAMPLE_RATE = 512
FFT_SIZE = 1024
BANDWIDTH = 20

def open_spectrogram(filename, fft_size, start_sample, num_samples):
    f = open(filename, "rb") 
    num_to_skip = fft_size * start_sample
    f.seek(num_to_skip, os.SEEK_SET)
    num_to_read = fft_size * num_samples
    
    data_in_numpy = np.fromfile(f, count=num_to_read, dtype=np.complex64)
    print num_to_read
    print data_in_numpy.shape
    data_in_numpy = data_in_numpy.reshape(num_samples, fft_size)

    # FFTShift the array
    shifted_array = np.fft.fftshift(data_in_numpy, axes=(1,))
    return shifted_array

def decimate_array(array, downsample):
    decimated_array = decimate(array, downsample, axis = 0)
    return decimated_array


def file_to_array(filename):
    start_time = 0

    start_sample = int(start_time*SAMPLE_RATE)
    duration = -1

    num_samples = int(duration*SAMPLE_RATE)
    
    fft_size = FFT_SIZE
    
    numpy_array = open_spectrogram(filename, fft_size, start_sample, num_samples)

    decimated_array = decimate_array(numpy_array, 1)

    return decimated_array
