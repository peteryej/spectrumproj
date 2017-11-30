#!/usr/bin/env python
from transmitters import transmitters
from source_alphabet import source_alphabet
import analyze_stats
from gnuradio import channels, gr, blocks, filter
import numpy as np
import numpy.fft, cPickle, gzip
from scipy import signal, fftpack
import random
import time
import os
import sys

sys.path.append("/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages")

'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True

dataset = {}
dataset['metadata'] = {}

# Make directory of dataset
dir_name = 'dataset_localized_7_singlesource'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Absolute parameters
SAMP_RATE = 1e6
SPECTOGRAM_HEIGHT = 128

# The output format looks like this
# {('mod type', SNR): np.array(nvecs_per_key, 2, vec_length), etc}

# CIFAR-10 has 6000 samples/class. CIFAR-100 has 600. Somewhere in there seems like right order of magnitude
num_samples = 40000
max_num_sources = 1
vec_length = SPECTOGRAM_HEIGHT
spec_vec_length = 112*vec_length + 112
snr_vals = range(-20,20,2)
# For testing, let's do a small dataset first
snr_vals = [10]



fft_bw = SAMP_RATE/2 - SAMP_RATE/vec_length - 20e3

# Approximate bandwidths of the signal types:

approx_bandwidths = {
    'BPSK':100e3,
    'QPSK':150e3,
    '8PSK':150e3,
    'PAM4':120e3,
    'QAM16':150e3,
    'QAM64':120e3,
    'GFSK':40e3,
    'CPFSK':100e3,
    #'WBFM':30e3,
    #'AM-DSB':20e3,
    #'AM-SSB':2*fft_bw
}


class SignalSource(gr.hier_block2):
    def __init__(self, mod, alphabet_type, freq_offset, tx_len = int(30e4), approx_bw = 20, index = 0):
        gr.hier_block2.__init__(self, "transmitter_" + str(index),
            gr.io_signature(0, 0, 0) ,
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = mod()
        self.freq_offset = freq_offset

        self.src = source_alphabet(alphabet_type, tx_len, True, False)
        self.approx_bw = approx_bw
        self.freq_shift_filter = filter.freq_xlating_fir_filter_ccc(1, ([1]), -1*self.freq_offset, SAMP_RATE)
        #self.connect(self, self.src, self.mod, self)
        self.connect(self.src, self.mod, (self.freq_shift_filter,0), self)

    """ Get the corresponding center, height, width given a 128x128 output image. We're also
    assuming that y = 64 is the center frequency."""
    def get_metadata(self):
        y_center = SPECTOGRAM_HEIGHT/2 - self.freq_offset/(SAMP_RATE/vec_length)
        x_center = SPECTOGRAM_HEIGHT/2 # We will put the position in the image later

        # Cap the y to the size of the spectogram
        y_center = y_center if y_center <= SPECTOGRAM_HEIGHT else SPECTOGRAM_HEIGHT
        y_center = y_center if y_center >= 0 else 0
        height = float(self.approx_bw)/(2*fft_bw) * SPECTOGRAM_HEIGHT
        width = vec_length # We will put the actual TX length in later

        return {'center':(x_center,y_center),
                'width': width,
                'height': height,
                'label': self.mod.modname}

""" Randomly generate a source. The source has a center frequency offset,
a modulation type (transmitter)"""
def generate_source(index):
    # Choose modulation type.
    discrete_sources = transmitters['discrete']
    cont_sources = transmitters['continuous']
    # Choose proportionally based on length of discrete and continuous arrays
    type_selector = random.random()

    # Create discrete source
    #freq_offset = 0
    if True:
        type_selector < float(len(discrete_sources))/(len(cont_sources)+len(discrete_sources)):
        # Choose mod type
        mod_type = random.choice(discrete_sources)
        approx_bw = approx_bandwidths[mod_type.modname]
        # Choose center frequency
        freq_offset = random.uniform(-fft_bw + approx_bw/2, fft_bw - approx_bw/2)
        signal_source = SignalSource(mod_type, 'discrete', freq_offset, approx_bw = approx_bw, index=index)
    else: # Create continuous source
        # Choose mod type
        mod_type = random.choice(cont_sources)
        approx_bw = approx_bandwidths[mod_type.modname]
        # Choose center frequency
        freq_offset = random.uniform(-fft_bw+approx_bw/2, fft_bw - approx_bw/2)
        signal_source = SignalSource(mod_type, 'continuous', freq_offset, approx_bw = approx_bw, index = index)
    print mod_type.modname
    print freq_offset
    print fft_bw

    return signal_source

def main():
    insufficient_samples = True
    modvec_indx = 0
    with open (os.path.join(dir_name, 'index.txt'), 'a') as index_file:
        for snr in snr_vals:
            print "snr is ", snr

            # Create subfolder
            # if snr < 0:
            #     folderpath = os.path.join(dir_name,"negative_"+str(abs(snr)))
            # else:
            #     folderpath = os.path.join(dir_name,str(snr))
            #

            while insufficient_samples:
                #   if mod_type.modname == "QAM16":
                #       tx_len = int(20e4)
                #   if mod_type.modname == "QAM64":
                #       tx_len = int(30e4)

                tx_len = int(30e4)
                fD = 1
                delays = [0.0, 0.9, 1.7]
                mags = [1, 0.8, 0.3]
                ntaps = 8
                noise_amp = 10**(-snr/10.0)
                chan = channels.dynamic_channel_model(SAMP_RATE, 0.01, 50, .01, 0.5e3, 8, fD, True, 4, delays, mags, ntaps, noise_amp, 0x1337 )

                tb = gr.top_block()

                # Choose 1, 2, or 3 sources
                num_sources = random.randint(1,max_num_sources)
                sources_list = []
                source_add_block = blocks.add_vcc(1)

                # Generate num_sources sources
                for i in range(0, num_sources):
                    src = generate_source(i)
                    tb.connect(src, (source_add_block,i))
                    sources_list.append(src)

                snk = blocks.vector_sink_c()

                # Do limiter here at the end, so that different sources don't blocks
                limiter = blocks.head(gr.sizeof_gr_complex, tx_len)

                # connect blocks
                if apply_channel:
                  tb.connect((source_add_block,0), chan, limiter, snk)
                else:
                  tb.connect((source_add_block,0), limiter, snk)

                # This is a really cheesy fix! Just run for a couple seconds then quit
                tb.start()
                time.sleep(1)
                tb.stop()

                raw_output_vector = np.array(snk.data(), dtype=np.complex64)
                # start the sampler some random time after channel model transients (arbitrary values here)
                sampler_indx = random.randint(50, 500)
                while sampler_indx + spec_vec_length < len(raw_output_vector) and modvec_indx < num_samples:
                    # Get a snippet of vector to turn into a spectogram
                    sampled_vector = raw_output_vector[sampler_indx:sampler_indx+spec_vec_length]

                    # Normalize the energy in this vector to be 1
                    #   energy = np.sum((np.abs(sampled_vector)))

                    # Normalize with the max
                    #   max_in_vec = np.amax(np.abs(sampled_vector))
                    #   sampled_vector = sampled_vector / max_in_vec
                    f, t, Sxx = signal.spectrogram(sampled_vector, fs = SAMP_RATE, nperseg=vec_length, return_onesided= False, mode = 'complex')

                    # By default, the ordering of the frequencies it non-obvious.
                    # Use FFTShift to get it to go negative/positive.
                    f = fftpack.fftshift(f)
                    Sxx = fftpack.fftshift(Sxx, axes=0)

                    datapoint = {}

                    datapoint['data'] = np.zeros([vec_length, vec_length, 2], dtype=np.float32)

                    # Write data to file
                    datapoint['data'][:,:,0] = np.real(Sxx)
                    datapoint['data'][:,:,1] = np.imag(Sxx)
                    #datapoint['raw_data'] = sampled_vector
                    datapoint['sample_f'] = f
                    datapoint['segment_t'] = t
                    datapoint['objects'] = [src.get_metadata() for src in sources_list]

                    filename = str(modvec_indx) + '.dat'

                    filepath = os.path.join(dir_name,filename)
                    cPickle.dump( datapoint, file(filepath, "wb" ) )

                    print datapoint['objects']

                    # Write metadata to text file
                    index_file.write(filename + ' ' + str(snr) + '\n')

                    #print Sxx.shape
                    #   dataset[(mod_type.modname, snr)][modvec_indx,0,:] = np.real(sampled_vector)
                    #   dataset[(mod_type.modname, snr)][modvec_indx,1,:] = np.imag(sampled_vector)
                    # bound the upper end very high so it's likely we get multiple passes through
                    # independent channels
                    sampler_indx += random.randint(vec_length, round(len(raw_output_vector)*.05))
                    modvec_indx += 1
                    print "Generated Sample " + str(modvec_indx)

                if modvec_indx == num_samples:
                    # we're all done
                    insufficient_samples = False

                  # Store sample frequencies and segment times
    print "all done. writing to disk"

if __name__ == '__main__':
    main()
