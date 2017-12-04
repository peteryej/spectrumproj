import binary_decoder
import matplotlib.pyplot as plt
import test_generate as gd
import cPickle
import os
import numpy as np

WINDOW = 10
PWD = os.getcwd()
DATAFOLDER = 'multi_channel_data'
OBJECTFOLDER = 'multi_channel_objects'
DATAFILE = 'data'
OBJFILE = 'object'
INPUTFILE = 'inputmultitraining.txt'
INDEXFILE = 'multi_channel_index.txt'

PIXEL_PER_SECOND = 128.0 / 10
PIXEL_PER_MH = 128.0 / 30

W = 128
H = 128

def sample(array):
    (h, w) = array.shape
    step_h = h / H
    step_w = w / W
    s = np.zeros((H, W), dtype=np.complex64)
    ch, cw = 0, 0
    for i in np.linspace(0, h - 1, num=H):
        for j in np.linspace(0, w - 1, num=W):
            x, y = int(i), int(j)
            s[ch, cw] = array[x, y]
            cw += 1
        cw = 0
        ch += 1

    return s

def generate_one_object(info):
    start = info['start']
    cf = info['cf']
    bw = info['bw']
    mod = info['mod']
    mod_num = info['mod_num']
    duration = info['duration']

    width = int(bw * PIXEL_PER_MH)
    height = int(duration * PIXEL_PER_SECOND)
    x_center = int((cf - 900) * PIXEL_PER_MH)
    y_center = int((start + duration / 2.0) * PIXEL_PER_SECOND)
    return {
        'center':(x_center,y_center),
        'width': width,
        'height': height,
        'label': mod + str(mod_num)
    }


if __name__ == '__main__':
    print("started")
    with open(INPUTFILE) as f:
        with open(INDEXFILE,'wb') as fi:
            for i, line in enumerate(f):
                info_list = gd.parse_a_line(line)
                if not any(info_list):
                    continue
                print('Generating object {}'.format(i))
                file_name = os.path.join(PWD,DATAFOLDER,DATAFILE+str(i)+'.dat')
                array = binary_decoder.file_to_array(file_name)
                (h, w) = array.shape
                array = sample(array)
                datapoint = {}
                datapoint['data'] = np.zeros((H,H,2), dtype=np.float32)
                datapoint['data'][:,:,0] = array.real
                datapoint['data'][:,:,1] = array.imag
                datapoint['objects'] = []
                for info in info_list:
                    if info is None:
                        continue
                    else:
                        datapoint['objects'].append(generate_one_object(info))

                save_file = os.path.join(PWD,OBJECTFOLDER,OBJFILE+str(i)+'.dat')
                fi.write('{} {}\n'.format(OBJFILE+str(i)+'.dat',0))
                print save_file
                cPickle.dump(datapoint, file(save_file, 'wb'))





def show(decimated_array):

   print "Dataloaded. figure outputs"
   plt.figure()
   plt.imshow(decimated_array, cmap = 'rainbow', extent=(-10, 10, 0, -1))
   plt.xlabel('Bandwidth (MHz)')
   plt.ylabel('Time (s)')

   plt.show()
