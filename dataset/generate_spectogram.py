#!/usr/bin/env python

import sys
sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')

from transmitters import transmitters
from source_alphabet import source_alphabet
import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, cPickle, gzip
from scipy import signal, fftpack
import random
import os



'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True

dataset = {}
dataset['metadata'] = {}

# Make directory of dataset
dir_name = 'dataset_4'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# The output format looks like this
# {('mod type', SNR): np.array(nvecs_per_key, 2, vec_length), etc}

# CIFAR-10 has 6000 samples/class. CIFAR-100 has 600. Somewhere in there seems like right order of magnitude
nvecs_per_key = 2
vec_length = 128
spec_vec_length = 112*vec_length + 112
snr_vals = range(-20,20,2)
# For testing, let's do a small dataset first
snr_vals = [10]


global_index = 0
with open (os.path.join(dir_name, 'index.txt'), 'a') as index_file:
    for snr in snr_vals:
        print "snr is ", snr

        # Create subfolder
        # if snr < 0:
        #     folderpath = os.path.join(dir_name,"negative_"+str(abs(snr)))
        # else:
        #     folderpath = os.path.join(dir_name,str(snr))
        #


        for alphabet_type in transmitters.keys():
            for i,mod_type in enumerate(transmitters[alphabet_type]):
                # cat_folderpath = os.path.join(folderpath,str(mod_type.modname))
                # if not os.path.exists(folderpath):
                #     os.makedirs(folderpath)
                #dataset[(mod_type.modname, snr)] = np.zeros([nvecs_per_key, vec_length, vec_length, 2], dtype=np.float32)
                # moar vectors!
                insufficient_modsnr_vectors = True
                modvec_indx = 0
                while insufficient_modsnr_vectors:
                  tx_len = int(10e5)
                #   if mod_type.modname == "QAM16":
                #       tx_len = int(20e4)
                #   if mod_type.modname == "QAM64":
                #       tx_len = int(30e4)
                  src = source_alphabet(alphabet_type, tx_len, True)
                  mod = mod_type()
                  fD = 1
                  delays = [0.0, 0.9, 1.7]
                  mags = [1, 0.8, 0.3]
                  ntaps = 8
                  noise_amp = 10**(-snr/10.0)
                  chan = channels.dynamic_channel_model( 200e3, 0.01, 50, .01, 0.5e3, 8, fD, True, 4, delays, mags, ntaps, noise_amp, 0x1337 )

                  snk = blocks.vector_sink_c()

                  tb = gr.top_block()

                  # connect blocks
                  if apply_channel:
                      tb.connect(src, mod, chan, snk)
                  else:
                      tb.connect(src, mod, snk)
                  tb.run()

                  raw_output_vector = np.array(snk.data(), dtype=np.complex64)
                  # start the sampler some random time after channel model transients (arbitrary values here)
                  sampler_indx = random.randint(50, 500)
                  while sampler_indx + spec_vec_length < len(raw_output_vector) and modvec_indx < nvecs_per_key:
                      # Get a snippet of vector to turn into a spectogram
                      sampled_vector = raw_output_vector[sampler_indx:sampler_indx+spec_vec_length]
                      # Normalize the energy in this vector to be 1
                    #   energy = np.sum((np.abs(sampled_vector)))

                      # Normalize with the max
                    #   max_in_vec = np.amax(np.abs(sampled_vector))
                    #   sampled_vector = sampled_vector / max_in_vec
                      f, t, Sxx = signal.spectrogram(sampled_vector, nperseg=vec_length, mode='complex')

                      # By default, the ordering of the frequencies it non-obvious.
                      # Use FFTShift to get it to go negative/positive.
                      f = fftpack.fftshift(f)
                      Sxx = fftpack.fftshift(Sxx, axes=0)

                      datapoint = {}

                      datapoint['data'] = np.zeros([vec_length, vec_length, 2], dtype=np.float32)

                      # Write data to file
                      datapoint['data'][:,:,0] = np.real(Sxx)
                      datapoint['data'][:,:,1] = np.imag(Sxx)
                      datapoint['sample_f'] = f
                      datapoint['segment_t'] = t

                      filename = str(global_index) + '.dat'
                      filepath = os.path.join(dir_name,filename)
                      cPickle.dump( datapoint, file(filepath, "wb" ) )

                      # Write metadata to text file
                      index_file.write(filename + ' ' + str(snr) + ' ' + str(mod_type.modname) + '\n')
                      global_index += 1

                      #print Sxx.shape
                    #   dataset[(mod_type.modname, snr)][modvec_indx,0,:] = np.real(sampled_vector)
                    #   dataset[(mod_type.modname, snr)][modvec_indx,1,:] = np.imag(sampled_vector)
                      # bound the upper end very high so it's likely we get multiple passes through
                      # independent channels
                      sampler_indx += random.randint(vec_length, round(len(raw_output_vector)*.05))
                      modvec_indx += 1

                  if modvec_indx == nvecs_per_key:
                      # we're all done
                      insufficient_modsnr_vectors = False

# Store sample frequencies and segment times

print "all done. writing to disk"
