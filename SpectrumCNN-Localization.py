
# coding: utf-8

# ## Modulation Recognition Example: RML2016.10a Dataset + VT-CNN2 Mod-Rec Network
# 
# More information on this classification method can be found at
# https://arxiv.org/abs/1602.04105
# 
# More information on the RML2016.10a dataset can be found at
# http://pubs.gnuradio.org/index.php/grcon/article/view/11
# 
# Please cite derivative works
# 
# ```
# @article{convnetmodrec,
#   title={Convolutional Radio Modulation Recognition Networks},
#   author={O'Shea, Timothy J and Corgan, Johnathan and Clancy, T. Charles},
#   journal={arXiv preprint arXiv:1602.04105},
#   year={2016}
# }
# @article{rml_datasets,
#   title={Radio Machine Learning Dataset Generation with GNU Radio},
#   author={O'Shea, Timothy J and West, Nathan},
#   journal={Proceedings of the 6th GNU Radio Conference},
#   year={2016}
# }
# ```
# 
# To run this example, you will need to download or generate the RML2016.10a dataset (https://radioml.com/datasets/)
# You will also need Keras installed with either the Theano or Tensor Flow backend working.
# 
# Have fun!

# In[1]:


# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
get_ipython().magic(u'matplotlib inline')
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
from keras.optimizers import adam, SGD

import matplotlib.pyplot as plt
#import seaborn as sns
import cPickle, random, sys, keras

import config


# # Test the datapoints to see if they're OK

# In[2]:


import matplotlib.pyplot as plt
import os
from scipy.fftpack import fft,fftshift
from scipy import signal
import matplotlib.ticker as ticker
from dataset.dataset_model import dataset_model

path = "dataset/dataset_localized_3/"

datapoint = cPickle.load(open(os.path.join(path,'216.dat')))

# plt.plot(np.real(datapoint['raw_data'][0:100]))
# plt.plot(np.imag(datapoint['raw_data'][0:100]))

# f, t, Sxx = signal.spectrogram(datapoint['raw_data'], nperseg=128)

# f = fftshift(f)
# Sxx = fftshift(Sxx, axes=0)

# plt.pcolormesh(t, f, np.abs(Sxx))

plt.figure(figsize=(8,6))
tick_spacing = 50000

# print datapoint['data'][:,:,0]

t = datapoint['segment_t']
f = datapoint['sample_f']

print datapoint['objects']

norm_data = dataset_model(path, localization_dataset = True)
datapoint_norm = norm_data.normalize_data(datapoint['data'])

matrix_data = datapoint['data']

print np.log10(matrix_data[:,:,0]/matrix_data[:,:,0].min())

norm_matrix_data = matrix_data * 1.0/matrix_data.max()

mag_data = np.sqrt(np.power(norm_matrix_data,2) + np.power(norm_matrix_data,2))

plt.pcolormesh(t,f, mag_data)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))


# print f


# # Create the dataset using generator

# In[3]:


from dataset.dataset_model import dataset_model

# Set up some params
nb_epoch = 100     # number of epochs to train on
batch_size = 64  # training batch size

train_set = dataset_model('dataset/dataset_localized_7_singlesource', localization_dataset = True)
test_set = dataset_model('dataset/dataset_localized_7_singlesource', localization_dataset = True)

train_set_gen = train_set.dataset_generator_localized(batch_size)
test_set_gen = test_set.dataset_generator_localized(batch_size)

num_classes = len(train_set.class_types)


# We also want to see how well we can classify on Fourier transforms of the signal. This will provide a clue as to how well it will work on Spectograms. 

# ## Let's try Tiny-Yolo
# 
# The Darknet Reference Model is commonly used in conjunction with the YOLO object detection methodology. It is significantly faster than GoogLeNet v1, which will be important for us. 

# In[4]:


import detregionloss
reload(detregionloss)

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

# Before this line is the Darknet Reference Model. After is the YOLO additions
# This is the OLD tiny-yolo model, circa mid-2016. We'll start with this and perhaps go to the
# new one later on. 

model.add(Conv2D(1024, conv_size, padding="same"))
#model.add(BatchNormalization(name='batchnorm8'))
model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(256, conv_size, padding="same"))
#model.add(BatchNormalization(name='batchnorm9'))
model.add(LeakyReLU(alpha=0.1))

model.add(Flatten())
model.add(Dense(4096))
#model.add(BatchNormalization(name='batchnorm10'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(dr, noise_shape=None, seed=None))
model.add(Dense(config.S_GRID**2*(5*config.B_BOXES+config.C_CLASS)))

sgd = SGD(lr=0.001, decay=0, momentum=0.9)
model.compile(loss=detregionloss.regionloss, optimizer=sgd, metrics = [])
print(model.summary())


# # Train the Model

# In[17]:


# Transfer learning. Import the previously trained model. 
model.load_weights('spectrumCNN.wts.h5', by_name=True)


# In[ ]:



# model.load_weights('spectrumCNN_Localized.wts.h5', by_name=True)
# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'spectrumCNN_Localized.wts.h5'
history = model.fit_generator(generator = train_set_gen,
    steps_per_epoch=int(10000/batch_size),
    epochs=200,
    verbose=2,
    validation_data = test_set_gen,
    validation_steps = 100,
    pickle_safe = True,
    workers = 8,                                        
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='auto')
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)


# # Try model on one data point

# In[59]:





# In[26]:


from utils import utils
reload(utils)
path = "dataset/dataset_localized_7_singlesource/"
filepath = 'spectrumCNN_Localized.wts.h5'
model.load_weights(filepath)

datapoint = cPickle.load(open(os.path.join(path,'7600.dat')))

# plt.plot(np.real(datapoint['raw_data'][0:100]))
# plt.plot(np.imag(datapoint['raw_data'][0:100]))

# f, t, Sxx = signal.spectrogram(datapoint['raw_data'], nperseg=128)

# f = fftshift(f)
# Sxx = fftshift(Sxx, axes=0)

# plt.pcolormesh(t, f, np.abs(Sxx))

plt.figure(figsize=(8,6))
tick_spacing = 50000

# print datapoint['data'][:,:,0]

t = datapoint['segment_t']
f = datapoint['sample_f']

print datapoint['objects']

mag_data = np.sqrt(np.power(datapoint['data'][:,:,0],2) + np.power(datapoint['data'][:,:,1],2))


print datapoint['data'].max()
print datapoint['data'].min()

norm_data = train_set.normalize_data(datapoint['data'])

plt.pcolormesh(t,f, mag_data)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

model_out = model.predict(np.array([norm_data]), batch_size=1)

# dsmodel = dataset_model('dataset/dataset_localized_1', localization_dataset = True)
# junk,matrix =  dsmodel.extract_localization_data('201.dat')

boxes =  utils.yolo_net_out_to_boxes(model_out, threshold=0.215)

for box in boxes:
    print config.CLASS_TYPES[box.label_index]
    print box.prob

mag_data_boxes = utils.draw_box(boxes, mag_data)

plt.figure()
plt.pcolormesh(t,f, mag_data_boxes)


# In[98]:



input_45 = tf.placeholder(tf.float32, (4))
output_45 = tf.nn.softmax(input_45)

sesh = tf.Session()
feed_dict45 = {input_45: np.array([1,1,50,1])}
sesh.run([output_45], feed_dict = feed_dict45)


# In[9]:


#Test loss function 
import detregionloss
import tensorflow as tf
reload(detregionloss)

datapoint = cPickle.load(open(os.path.join(path,'213.dat')))
print datapoint['objects']
junk,matrixtest =  dsmodel.extract_localization_data('213.dat')

input_node = tf.placeholder(tf.float32, np.array([matrixtest]).shape)

loss =  detregionloss.yololoss(input_node, input_node)

feed_dict = {input_node: np.array([matrixtest])}

sesh = tf.Session()
sesh.run([loss], feed_dict=feed_dict)
#sesh.close()


# # Evaluate and Plot Model Performance

# In[13]:


# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print score


# In[19]:


# Show loss curves 
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.plot(history.epoch, history.history['val_acc'], label='val_accuracy')
plt.legend()


# In[92]:


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[93]:


# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)



# In[94]:


# Plot confusion matrix
acc = {}
for snr in snrs:
    print snr

    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print "Overall Accuracy: ", cor / (cor+ncor)
    acc[snr] = 1.0*cor/(cor+ncor)
    


# In[13]:


# Save results to a pickle file for plotting later
print acc
fd = open('results_cnn2_d0.5.dat','wb')
cPickle.dump( ("CNN2", 0.5, acc) , fd )


# In[95]:


# Plot accuracy curve
plt.plot(snrs, map(lambda x: acc[x], snrs))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")

