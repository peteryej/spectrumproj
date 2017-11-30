
# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
#get_ipython().magic(u'matplotlib inline')
import os,random
#os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Permute,Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, Conv1D, MaxPooling2D, ZeroPadding2D, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
#import seaborn as sns
import cPickle, random, sys, keras



# # Test the datapoints to see if they're OK

# In[2]:


import matplotlib.pyplot as plt
import os

path = "newdataset/"

datapoint = cPickle.load(open(os.path.join(path,'object_generated0.dat')))

#print datapoint['data'][:,:,0]
print datapoint['data'][:,:]

#t = datapoint['segment_t']
#f = datapoint['sample_f']
t = 1024;
f = 293;

plt.pcolormesh(datapoint['data'][:,:])






num_classes = 1


dr = 0.5
conv_size = (3,3)
model = models.Sequential()
model.add(Conv2D(16, conv_size, padding="same", strides=(1, 1), input_shape=(128, 128,2), name='conv1'))
#model.add(BatchNormalization(name='batchnorm1')) # We don't add any axis as last is correct 
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, conv_size, padding="same", name='conv2'))
#model.add(BatchNormalization(name='batchnorm2'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))

model.add(Conv2D(64, conv_size, padding="same", name='conv3'))
#model.add(BatchNormalization(name='batchnorm3'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))

model.add(Conv2D(128, conv_size, padding="same", name='conv4'))
#model.add(BatchNormalization(name='batchnorm4'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))

model.add(Conv2D(256, conv_size, padding="same", name='conv5'))
#model.add(BatchNormalization(name='batchnorm5'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))

model.add(Conv2D(512, conv_size, padding="same", name='conv6'))
#model.add(BatchNormalization(name='batchnorm6'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))

model.add(Conv2D(1024, conv_size, padding="same", name='conv7'))
#model.add(BatchNormalization(name='batchnorm7'))
model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(num_classes, conv_size, padding="same", name='conv8'))
model.add(GlobalAveragePooling2D())

model.add(Activation('softmax'))
model.add(Reshape([num_classes]))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
