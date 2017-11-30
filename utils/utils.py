# code based on:
# CarND Vehicle Detection (xslittlegrass) https://github.com/xslittlegrass/CarND-Vehicle-Detection

import numpy as np
import cv2
import config

def load_weights(model,yolo_weight_file):

    data = np.fromfile(yolo_weight_file,np.float32)
    data=data[4:]

    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            kshape,bshape = shape
            bia = data[index:index+np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index+np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker,bia])


class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()
        self.label_index = int()

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);

# This really should be in Tensorflow. Softmax layer needs to be part of the
# network itself
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def yolo_net_out_to_boxes(net_out, class_num=0, threshold = 0.2):
    C = config.C_CLASS
    B = config.B_BOXES
    S = config.S_GRID
    boxes = []
    SS        =  S * S # number of grid cells
    prob_size = SS * C # class probabilities
    conf_size = SS * B # confidences for each grid cell


    reshape_out = np.swapaxes(np.reshape(net_out, (7,7,21), order = 'F'), 0,1)

    confs = np.stack((reshape_out[:,:,4], reshape_out[:,:,9]), axis=2)
    probs = reshape_out[:,:,B*5 : ]
    coords =  np.concatenate((reshape_out[:,:,0:4], reshape_out[:,:,5:9]), axis=2)
    print coords.shape
    # confs = net_out[prob_size : (prob_size + conf_size)]
    #cords = net_out[(prob_size + conf_size) : ]

    # probs = probs.reshape([SS, C])
    # confs = confs.reshape([SS, B])
    # cords = cords.reshape([SS, B, 4])

    for grid_x in range(0, S):
        for grid_y in range(0, S):
            for grid_b in range(0,B):
                bx   = Box()
                bx.c = confs[grid_y, grid_x, grid_b]
                bx.x = (coords[grid_y, grid_x, grid_b*4] + grid_x)/S
                bx.y = (coords[grid_y, grid_x, grid_b*4 + 1] + grid_y)/S
                bx.w = coords[grid_y, grid_x, grid_b*4 + 2]
                bx.h = coords[grid_y, grid_x, grid_b*4 + 3]
                p = softmax(probs[grid_y, grid_x, :]) * bx.c

                if np.amax(p) >= threshold:
                    bx.prob = np.amax(p)
                    bx.label_index = np.argmax(p)
                    boxes.append(bx)


    # for grid in range(SS):
    #     for b in range(B):
    #         bx   = Box()
    #         bx.c =  confs[grid, b]
    #         bx.x = (cords[grid, b, 0] + grid %  S) / S
    #         bx.y = (cords[grid, b, 1] + grid // S) / S
    #         bx.w =  cords[grid, b, 2] ** sqrt
    #         bx.h =  cords[grid, b, 3] ** sqrt
    #         p = probs[grid, :] * bx.c
    #
    #         if p[class_num] >= threshold:
    #             bx.prob = p[class_num]
    #             boxes.append(bx)

    # combine boxes that are overlap
    print boxes
    # boxes.sort(key=lambda b:b.prob,reverse=True)
    # for i in range(len(boxes)):
    #     boxi = boxes[i]
    #     if boxi.prob == 0: continue
    #     for j in range(i + 1, len(boxes)):
    #         boxj = boxes[j]
    #         if box_iou(boxi, boxj) >= .4:
    #             boxes[j].prob = 0.
    # boxes = [b for b in boxes if b.prob > 0.]

    print boxes

    return boxes

def draw_box(boxes,im):
    imgcv = np.zeros((config.IMG_H, config.IMG_W))
    [xmin,xmax] = [0, config.IMG_W]
    [ymin,ymax] = [0, config.IMG_H]
    for b in boxes:
        h = config.IMG_H
        w = config.IMG_W
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        left = int(left*(xmax-xmin)/w + xmin)
        right = int(right*(xmax-xmin)/w + xmin)
        top = int(top*(ymax-ymin)/h + ymin)
        bot = int(bot*(ymax-ymin)/h + ymin)

        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        thick = int((h + w) // 150)
        cv2.rectangle(imgcv, (xmax- left, ymax - top), (xmax - right, ymax - bot), (255,0,0), thick)

    return imgcv
