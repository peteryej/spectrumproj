import os
import random
import cPickle
import numpy as np
import time
import config
from utils import utils
import threading

# from transmitters import transmitters

# from keras.utils.np_utils import to_categorical
# generators in multi-threaded applications is not thread-safe. Hence below:
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class dataset_model:
    def __init__(self, folder_name, localization_dataset = False):
        self.folder_name = folder_name
        self.is_localized = localization_dataset
        self.datapoints = self.get_by_dataset('index.txt')
        self.class_types = config.CLASS_TYPES
        if localization_dataset:
            self.truth_size = config.S_GRID**2*(5*config.B_BOXES+config.C_CLASS)

    """
    Opens a text file with the file names, called file_index.
    """
    def get_by_dataset(self, file_index):
        filepath = os.path.join(self.folder_name, file_index)
        datapoints = []
        with open(filepath) as f:
            for line in f:
                split_line = line.split()
                if not self.is_localized:
                    datapoint = {'filename': split_line[0],
                                'snr': int(split_line[1]),
                                'label' : split_line[2]}
                else:
                    datapoint = {'filename': split_line[0],
                                'snr': int(split_line[1])}
                datapoints.append(datapoint)
        random.shuffle(datapoints)
        return datapoints

    def get_data_from_filename(self, filename):
        filepath = os.path.join(self.folder_name, filename)
        datapoint = cPickle.load(open(filepath))
        return datapoint

    def to_onehot(self,yy):
        yy1 = np.zeros([len(yy), len(self.class_types)])
        yy1[np.arange(len(yy)),yy] = 1
        return yy1

    def extract_only_matrix_data(self, filename):
        datapoint = self.get_data_from_filename(filename)
        matrix_data = datapoint['data']
        return self.normalize_data(matrix_data)

    def normalize_data(self,matrix_data):
        #matrix_data_db = np.log10(matrix_data/matrix_data.min())
        norm_matrix_data = matrix_data/ np.abs(matrix_data).max()

        return norm_matrix_data

    def extract_localization_data(self, filename):
        datapoint = self.get_data_from_filename(filename)
        matrix_data = datapoint['data']
        objects = datapoint['objects']

    	grid_side = config.S_GRID
    	bnumPercell = config.B_BOXES
    	classes = config.C_CLASS

        truth = np.zeros(self.truth_size)

        ## Array Layout:
        # [Box 1(x,y,w,h,conf) (7x7x5)] [Box 2(x,y,w,h,conf)(7x7x5)] [Class Probs (7x7xC_CLASS)]

        for item in objects:
            box_x = float(item['center'][0])/config.IMG_W
            box_y = float(item['center'][1])/config.IMG_H
            box_h = float(item['height'])/config.IMG_H
            box_w = float(item['width'])/config.IMG_W


            col = int(box_x * grid_side)
            row = int(box_y * grid_side)
            x = box_x * grid_side - col
            y = box_y * grid_side - row

            # print col,row

            # 2 boxes for the cell
            box_offset = (grid_side**2 * 5)
            grid_index = (col+row*grid_side) # where within 7x7 grid (49-element array) you fall
            # col changes fastest

            for i in range(bnumPercell):
            	index = grid_index + i*box_offset
            	truth[index] = x
            	truth[grid_side**2+index] = y
            	truth[2*(grid_side**2)+index] = box_w
            	truth[3*(grid_side**2)+index] = box_h
            	truth[4*(grid_side**2)+index] = 1 # Confidence

            # Assign class marker
            class_id = self.class_types.index(item['label'])
            truth[(class_id)*(grid_side**2)+grid_index + 2*box_offset] = 1

        return self.normalize_data(matrix_data), truth

    @threadsafe_generator
    def dataset_generator(self, batch_size):
        start_index = 0
        end_index = 0

        # Because this might be used in a worker thread, we don't want it to mutate any common variables
        datapoints_copy = list(self.datapoints)
        random.shuffle(datapoints_copy)

        while True:
            # Initialize the batch as zeros
            batch_data = np.zeros((batch_size, config.IMG_H, config.IMG_W, 2))
            # Grab the first batch_size datapoints. If the index exceeds the end, re-shuffle
            # and re-start.
            start_index = end_index
            end_index = start_index + batch_size
            if end_index > len(datapoints_copy):
                print "Reached end of Set 1, reshuffling..."
                random.shuffle(datapoints_copy)
                start_index = 0
                end_index = start_index + batch_size
            # Get data as lists
            batch_filenames = [point['filename'] for point in datapoints_copy[start_index:end_index]]

            # Put images in batch_data
            for i in range(batch_size):
                batch_data[i,:,:,:] = self.extract_only_matrix_data(batch_filenames[i])
            batch_labels = [self.class_types.index(point['label']) for point in datapoints_copy[start_index:end_index]]

            batch_labels_onehot = self.to_onehot(batch_labels)
            #print batch_labels
            yield (batch_data, batch_labels_onehot)
    @threadsafe_generator
    def dataset_generator_localized(self, batch_size):
        start_index = 0
        end_index = 0

        # Because this might be used in a worker thread, we don't want it to mutate any common variables
        datapoints_copy = list(self.datapoints)
        random.shuffle(datapoints_copy)


        while True:
            # Initialize the batch as zeros
            batch_data = np.zeros((batch_size, config.IMG_H, config.IMG_W, 2))
            batch_labels = np.zeros((batch_size, self.truth_size))
            # Grab the first batch_size datapoints. If the index exceeds the end, re-shuffle
            # and re-start.
            start_index = end_index
            end_index = start_index + batch_size
            if end_index > len(datapoints_copy):
                print "Reached end of Set 1, reshuffling..."
                random.shuffle(datapoints_copy)
                start_index = 0
                end_index = start_index + batch_size
            # Get data as lists
            batch_filenames = [point['filename'] for point in datapoints_copy[start_index:end_index]]

            # Put images in batch_data
            for i in range(batch_size):
                img_data, label_data = self.extract_localization_data(batch_filenames[i])
                batch_data[i,:,:,:] = img_data
                batch_labels[i,:] = label_data

            yield (batch_data, batch_labels)


if __name__ == '__main__':
    dsmodel = dataset_model('dataset/dataset_localized_1', localization_dataset = True)
    junk,matrix =  dsmodel.extract_localization_data('213.dat')

    reshape_matrix = np.reshape(matrix, (7,7,21), order = 'F')

    print utils.yolo_net_out_to_boxes(matrix)

    # for i in range(0,21):
    #     print reshape_matrix[:,:,i]
    # gen = dsmodel.dataset_generator_localized(10)
    # print next(gen)[1].shape



    # while True:
    #     print len(next(gen)[1])
