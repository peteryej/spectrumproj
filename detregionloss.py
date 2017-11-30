from keras import backend as K
import tensorflow as tf
import numpy as np
import config


#
side = config.S_GRID
gridcells = config.S_GRID**2
lamda_confid_obj =  config.LAMBDA_OBJ
lamda_confid_noobj = config.LAMBDA_NOOBJ
lamda_xy = config.COORD_SCALE
lamda_wh = config.COORD_SCALE
reguralar_wh = 0
lamda_class = config.CLASS_SCALE
classes = config.NUM_CLASS
b_boxes = config.B_BOXES

DEBUG_loss = False

# shape is (gridcells,)
def yoloconfidloss(y_true, y_pred, t):
	# Why is it sigmoid?
	# pobj = K.sigmoid(y_pred)
	lo = K.square(y_true-y_pred)
	value_if_true = lamda_confid_obj*(lo)
	value_if_false = lamda_confid_noobj*(lo)
	loss1 = tf.where(t, value_if_true, value_if_false)
	loss = K.mean(loss1) #,axis=0)
	#
	ave_anyobj = K.mean(y_pred)
	obj = tf.where(t, y_pred, K.zeros_like(y_pred))
	objcount = tf.where(t, K.ones_like(y_pred), K.zeros_like(y_pred))
	ave_obj = K.mean( K.sum(obj, axis=1) / (K.sum(objcount, axis=1)+0.000001) ) # prevent div 0
	return loss, ave_anyobj, ave_obj

# shape is (gridcells*2,)
def yoloxyloss(y_true, y_pred, t):
	lo = K.square(y_true-y_pred)
	value_if_true = lamda_xy*(lo)
	value_if_false = K.zeros_like(y_true)
	loss1 = tf.where(t, value_if_true, value_if_false)
	return K.mean(loss1)

# different with YOLO
# shape is (gridcells*2,)
def yolowhloss(y_true, y_pred, t):
	lo = K.square(K.sqrt(y_true)-K.sqrt(y_pred))
	# let w,h not too small or large
	#lo = K.square(y_true-y_pred)+reguralar_wh*K.square(0.5-y_pred)
	value_if_true = lamda_wh*(lo)
	value_if_false = K.zeros_like(y_true)
	loss1 = tf.where(t, value_if_true, value_if_false)
	#return K.mean(loss1/(y_true+0.000000001))
	return K.mean(loss1)

# shape is (gridcells*classes,)
def yoloclassloss(y_true, y_pred, t):
	lo = K.square(y_true-y_pred)
	value_if_true = lamda_class*(lo)
	value_if_false = K.zeros_like(y_true)
	loss1 = tf.where(t, value_if_true, value_if_false)
	# only extract predicted class value at obj location
	cat = K.sum(tf.where(t, y_pred, K.zeros_like(y_pred)), axis=1)
	# check valid class value
	objsum = K.sum(y_true, axis=1)
	# if objsum > 0.5 , means it contain some valid obj(may be 1,2.. objs)
	isobj = K.greater(objsum, 0.5)
	# only extract class value at obj location
	valid_cat = tf.where(isobj, cat, K.zeros_like(cat))
	# prevent div 0
	ave_cat = tf.where(K.greater(K.sum(objsum),0.5), K.sum(valid_cat) / K.sum(objsum) , -1)
	return K.mean(loss1), ave_cat

def overlap(x1, w1, x2, w2):
	l1 = (x1) - w1/2
	l2 = (x2) - w2/2
	left = tf.where(K.greater(l1,l2), l1, l2)
	r1 = (x1) + w1/2
	r2 = (x2) + w2/2
	right = tf.where(K.greater(r1,r2), r2, r1)
	result = right - left
	return result

def iou(x_true,y_true,w_true,h_true,x_pred,y_pred,w_pred,h_pred,t):
	xoffset = K.cast_to_floatx((np.tile(np.arange(side),side)))
	yoffset = K.cast_to_floatx((np.repeat(np.arange(side),side)))
	# x = tf.where(t, K.sigmoid(x_pred), K.zeros_like(x_pred))
	# y = tf.where(t, K.sigmoid(y_pred), K.zeros_like(y_pred))
	# w = tf.where(t, K.sigmoid(w_pred), K.zeros_like(w_pred))
	# h = tf.where(t, K.sigmoid(h_pred), K.zeros_like(h_pred))
	# Again, why is it sigmoid?
	x = tf.where(t, x_pred, K.zeros_like(x_pred))
	y = tf.where(t, y_pred, K.zeros_like(y_pred))
	w = tf.where(t, w_pred, K.zeros_like(w_pred))
	h = tf.where(t, h_pred, K.zeros_like(h_pred))
	ow = overlap(x+xoffset, w*side, x_true+xoffset, w_true*side)
	oh = overlap(y+yoffset, h*side, y_true+yoffset, h_true*side)
	ow = tf.where(K.greater(ow,0), ow, K.zeros_like(ow))
	oh = tf.where(K.greater(oh,0), oh, K.zeros_like(oh))
	intersection = ow*oh
	union = w*h*(side**2) + w_true*h_true*(side**2) - intersection + K.epsilon()  # prevent div 0
	#
	recall_iou = intersection / union
	recall_t = K.greater(recall_iou, 0.5)
	recall_count = K.sum(tf.where(recall_t, K.ones_like(recall_iou), K.zeros_like(recall_iou)))
	#
	iou = K.sum(intersection / union, axis=1)
	obj_count = K.sum(tf.where(t, K.ones_like(x_true), K.zeros_like(x_true)) )
	ave_iou = K.sum(iou) / (obj_count)
	recall = recall_count / (obj_count)
	return ave_iou, recall, obj_count, intersection, union,ow,oh,x,y,w,h

# shape is (gridcells*(5+classes), )
def yololoss(y_true, y_pred):
	truth_confid_tf = tf.slice(y_true, [0,gridcells*4], [-1,gridcells])
	truth_x_tf = tf.slice(y_true, [0,0], [-1,gridcells])
	truth_y_tf = tf.slice(y_true, [0,gridcells], [-1,gridcells])
	truth_w_tf = tf.slice(y_true, [0,gridcells*2], [-1,gridcells])
	truth_h_tf = tf.slice(y_true, [0,gridcells*3], [-1,gridcells])

	truth_classes_tf = []
	for i in range(classes):
		ctf = tf.slice(y_true, [0,gridcells*(5*b_boxes+i)], [-1,gridcells])
		truth_classes_tf.append(ctf)


	pred_confid_tf = tf.slice(y_pred, [0,gridcells*4], [-1,gridcells])
	pred_x_tf = tf.slice(y_pred, [0,0], [-1,gridcells])
	pred_y_tf = tf.slice(y_pred, [0,gridcells], [-1,gridcells])
	pred_w_tf = tf.slice(y_pred, [0,gridcells*2], [-1,gridcells])
	pred_h_tf = tf.slice(y_pred, [0,gridcells*3], [-1,gridcells])

	#
	# below transformation is for softmax calculate
	# slice classes parta, shape is (samples, classes for one sample)
	classall = tf.slice(y_pred, [0,gridcells*5*b_boxes], [-1,gridcells*classes])
	# shape (samples, class for one sample) --> shape (samples, gridcells rows, classes cols)
	# every row contain 1 class with all cells
	classall_celltype = K.reshape(classall, (-1, gridcells, classes))
	# transpose shape to (samples, gridcells rows, classes cols)
	# this is for softmax operation shape
	# every row contain all classes with 1 cell
	#classall_softmaxtype = tf.transpose(classall_celltype, perm=(0,2,1))  # backend transpose function didnt support this kind of transpose
	# doing softmax operation, shape is (samples, gridcells rows, classes cols)
	class_softmax_softmaxtype = tf.nn.softmax(classall_celltype, dim = -1)
	# transpose back to shape (samples, classes rows, gridcells cols)
	#classall_softmax_celltype = tf.transpose(class_softmax_softmaxtype, perm=(0,2,1))  # backend transpose function didnt support this kind of transpose
	# change back to original matrix type,  but with softmax value
	pred_classall_softmax_tf = K.reshape(class_softmax_softmaxtype, (-1, classes*gridcells))

	#return classall, classall_celltype, classall_softmaxtype, class_softmax_softmaxtype, classall_softmax_celltype, pred_classall_softmax_tf
	pred_classes_tf = []
	for i in range(classes):
		#ctf = tf.slice(y_pred, [0,gridcells*(5+i)], [-1,gridcells])
		ctf = tf.slice(pred_classall_softmax_tf, [0,gridcells*(0+i)], [-1,gridcells])
		pred_classes_tf.append(ctf)

	t = K.greater(truth_confid_tf, config.CONF_THRESH)

	confidloss, ave_anyobj, ave_obj = yoloconfidloss(truth_confid_tf, pred_confid_tf, t)
	xloss = yoloxyloss(truth_x_tf, pred_x_tf, t)
	yloss = yoloxyloss(truth_y_tf, pred_y_tf, t)
	wloss = yolowhloss(truth_w_tf, pred_w_tf, t)
	hloss = yolowhloss(truth_h_tf, pred_h_tf, t)



	ave_iou, recall,obj_count, intersection, union,ow,oh,x,y,w,h = iou(truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t)

	classesloss =0.
	ave_cat =0.
	count =0.
	#closslist = []
	#catlist = []
	for i in range(classes):
		closs, cat = yoloclassloss(truth_classes_tf[i], pred_classes_tf[i], t)
		#closslist.append(closs)
		#catlist.append(cat)
		classesloss += closs
		ave_cat = tf.where(K.greater(cat ,0), (ave_cat+cat) , ave_cat)
		count = tf.where(K.greater(cat ,0), (count+1.) , count)
	ave_cat = ave_cat / count

	#return classesloss, ave_cat

	loss = confidloss+xloss+yloss+wloss+hloss+classesloss
	#loss = wloss+hloss
	#
	return loss,confidloss,xloss,yloss,wloss,hloss,classesloss, ave_cat, ave_obj, ave_anyobj, ave_iou, recall,obj_count, intersection, union,ow,oh,x,y,w,h
	#return loss, ave_cat, ave_obj, ave_anyobj, ave_iou


def limit(x):
	y = tf.where(K.greater(x,100000), 1000000.*K.ones_like(x), x)
	z = tf.where(K.less(y,-100000), -1000000.*K.ones_like(x), y)
	return z

def regionloss(y_true, y_pred):
	limited_pred = limit(y_pred)
	loss,confidloss,xloss,yloss,wloss,hloss,classesloss, ave_cat, ave_obj, ave_anyobj, ave_iou, recall,obj_count, intersection, union,ow,oh,x,y,w,h = yololoss(y_true, limited_pred)
	#return confidloss+xloss+yloss+wloss+hloss
	return loss

def regionmetrics(y_true, y_pred):
	limited_pred = limit(y_pred)
	loss,confidloss,xloss,yloss,wloss,hloss,classesloss, ave_cat, ave_obj, ave_anyobj, ave_iou, recall,obj_count, intersection, union,ow,oh,x,y,w,h = yololoss(y_true, limited_pred)
	pw = K.sum(w)
	ph = K.sum(h)
	return {
		#'loss' : loss,
		#'confidloss' : confidloss,
		#'xloss' : xloss,
		#'yloss' : yloss,
		#'wloss' : wloss,
		#'hloss' : hloss,
		#'classesloss' : classesloss,
		'ave_cat' : ave_cat,
		'ave_obj' : ave_obj,
		'ave_anyobj' : ave_anyobj,
		'ave_iou' : ave_iou,
		'recall' : recall,
		'obj_count' : obj_count
		#'predw' : pw,
		#'predh' : ph,
		#'ow' : K.sum(ow),
		#'oh' : K.sum(oh),
		#'insec' : K.sum(intersection),
		#'union' : K.sum(union)
	}


def check(detection_layer,model):
    expected = gridcells*(5+classes)
    real = model.layers[len(model.layers)-1].output_shape[1]
    if expected != real:
        print 'cfg detection layer setting mismatch::change cfg setting'
        print 'output layer should be '+str(expected)+'neurons'
        print 'actual output layer is '+str(real)+'neurons'
        exit()

#
#
if DEBUG_loss:

	side = 5
	obj_row = 2
	obj_col = 2
	obj_class = 6

	x_true =K.placeholder(ndim=2)
	x_pred =K.placeholder(ndim=2)
	#classall, classall_celltype, classall_softmaxtype, class_softmax_softmaxtype, classall_softmax_celltype, pred_classall_softmax_t = yololoss(x_true, x_pred)
	classesloss, ave_cat = yololoss(x_true, x_pred)
	#classcheck_f = K.function([x_true, x_pred], [classall, classall_celltype, classall_softmaxtype, class_softmax_softmaxtype, classall_softmax_celltype, pred_classall_softmax_t])
	classcheck_f = K.function([x_true, x_pred], [classesloss, ave_cat])
	tx = np.zeros((side**2)*(classes+5))
	tx[side*obj_row+obj_col] = 1
	tx[(side**2)*(5+obj_class)+side*obj_row+obj_col] = 1

	px = np.arange((side**2)*(classes+5))

	#a0,a1,a2,a3,a4,a5 = classcheck_f([np.asarray([tx]),np.asarray([px])])
	a0,a1 = classcheck_f([np.asarray([tx]),np.asarray([px])])
	print a0

        #t =K.placeholder(ndim=2, dtype=tf.bool)
        #truth_x_tf =K.placeholder(ndim=2)
        #truth_y_tf =K.placeholder(ndim=2)
        #truth_w_tf =K.placeholder(ndim=2)
        #truth_h_tf =K.placeholder(ndim=2)
        #pred_x_tf =K.placeholder(ndim=2)
        #pred_y_tf =K.placeholder(ndim=2)
        #pred_w_tf =K.placeholder(ndim=2)
        #pred_h_tf =K.placeholder(ndim=2)

        #ave_iou,recall, intersection, union,ow,oh,x,y,w,h = iou(truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t)
	#iouf = K.function([truth_x_tf,truth_y_tf,truth_w_tf,truth_h_tf,pred_x_tf,pred_y_tf,pred_w_tf,pred_h_tf,t], [ave_iou,recall,obj_count, intersection, union,ow,oh,x,y,w,h])
	# 0.507 0.551051051051 0.39 0.51951951952
	#np_t = np.zeros((side**2)*2).reshape(2,side**2)
	#obj_t = np_t >1
	#obj_t[0][obj_row*side+obj_col] = True
	#obj_t[1][obj_row*side+obj_col] = True
	#tx = np.zeros((side**2)*2).reshape(2,side**2)
	#ty = np.zeros((side**2)*2).reshape(2,side**2)
	#tw = np.zeros((side**2)*2).reshape(2,side**2)
	#th = np.zeros((side**2)*2).reshape(2,side**2)
	#tx[0][obj_row*side+obj_col] = 0.507*side - int(0.507*side)
	#ty[0][obj_row*side+obj_col] = 0.551051051051*side - int(0.551051051051*side)
	#tw[0][obj_row*side+obj_col] = 0.39
	#th[0][obj_row*side+obj_col] = 0.51951951952
	#px = np.random.random((side**2)*2).reshape(2,side**2)
	#py = np.random.random((side**2)*2).reshape(2,side**2)
	#pw = np.random.random((side**2)*2).reshape(2,side**2)
	#ph = np.random.random((side**2)*2).reshape(2,side**2)
	#px[0][obj_row*side+obj_col] = 0.5
	#py[0][obj_row*side+obj_col] = 0.5
	#pw[0][obj_row*side+obj_col] = 0.39/0.66
	#ph[0][obj_row*side+obj_col] = 0.51951951952/0.66

	#tx[1][obj_row*side+obj_col] = tx[0][obj_row*side+obj_col]
	#ty[1][obj_row*side+obj_col] = ty[0][obj_row*side+obj_col]
	#tw[1][obj_row*side+obj_col] = tw[0][obj_row*side+obj_col]
	#th[1][obj_row*side+obj_col] = th[0][obj_row*side+obj_col]
        #px[1][obj_row*side+obj_col] = px[0][obj_row*side+obj_col]
        #py[1][obj_row*side+obj_col] = py[0][obj_row*side+obj_col]
        #pw[1][obj_row*side+obj_col] = pw[0][obj_row*side+obj_col]
        #ph[1][obj_row*side+obj_col] = ph[0][obj_row*side+obj_col]


	#[a0,a1,a2,b0,b1,c0,c1,c2,c3]= iouf([tx,ty,tw,th,px,py,pw,ph,obj_t])
	#print a0


	#x =K.placeholder(ndim=2)
	#y =K.placeholder(ndim=2)
	#loss,confidloss,xloss,yloss,wloss,hloss,classesloss = yololoss(y,x)

	#f = K.function([y,x], [loss,confidloss,xloss,yloss,wloss,hloss,classesloss])

	#xtrain = np.ones(343*10).reshape(10,343)
	#ytrain = np.zeros(343*10).reshape(10,343)
	#ytrain[0][0]=1
	#ytrain[0][49]=0.1
	#ytrain[0][49*2]=0.2
	#ytrain[0][49*3]=0.3
	#ytrain[0][49*4]=0.4
	#ytrain[0][49*5]=1


	#print f([ytrain,xtrain])
