import numpy as np
from mxnet import gluon, nd
from mxnet.gluon import data as gdata, model_zoo, utils as gutils
import os

# this script was extract from my jupyter notebook. there might be some error. 
# I will update it later.

def normalizeFeature(x):
	# x = d x N dims (d: feature dimension, N: the number of features)
	x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide
	feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	feat = x / feature_norm[:, np.newaxis]
	return feat

def SAE(x, s, ld):
	# SAE is Semantic Autoencoder
	# INPUTS:
	# 	x: d x N data matrix
	#	s: k x N semantic matrix
	#	ld: lambda for regularization parameter
	#
	# OUTPUT:
	#	w: kxd projection matrix

	A = np.dot(s, s.transpose())
	B = ld * np.dot(x, x.transpose())
	C = (1+ld) * np.dot(s, x.transpose())
	w = scipy.linalg.solve_sylvester(A,B,C)
	return w

def distCosine(x, y):
	xx = np.sum(x**2, axis=1)**0.5
	x = x / xx[:, np.newaxis]
	yy = np.sum(y**2, axis=1)**0.5
	y = y / yy[:, np.newaxis]
	dist = 1 - np.dot(x, y.transpose())
	return dist

def zsl_acc(semantic_predicted, semantic_gt, final_classes_id, final_test_labels):
	# zsl_acc calculates zero-shot classification accruacy
	#
	# INPUTS:
	#	semantic_prediced: predicted semantic labels
	# 	semantic_gt: ground truth semantic labels
	# 	final_class_id: class ids that used for test
    #   final_test_labels: all labels for classification (not need to be test labels)
	#
	# OUTPUT:
	# 	zsl_accuracy: zero-shot classification accuracy (per-sample)

	dist = 1 - distCosine(semantic_predicted, normalizeFeature(semantic_gt.transpose()).transpose())
	y_hit_k = np.zeros((dist.shape[0], HITK))
	for idx in range(0, dist.shape[0]):
		sorted_id = sorted(range(len(dist[idx,:])), key=lambda k: dist[idx,:][k], reverse=True)
		y_hit_k[idx,:] = final_classes_id[sorted_id[0:HITK]]
		
	n = 0
	for idx in range(0, dist.shape[0]):
		if final_test_labels[idx] in y_hit_k[idx,:]:
			n = n + 1
	zsl_accuracy = float(n) / dist.shape[0] * 100
	return zsl_accuracy, y_hit_k

# split train and test images based on trainclasses.txt and testclasses.txt provided by AwA2
train_imgs = gdata.vision.ImageFolderDataset('../data/AwA2/Animals_with_Attributes2/JPEGImages/train/')
test_imgs = gdata.vision.ImageFolderDataset('../data/AwA2/Animals_with_Attributes2/JPEGImages/test/')

# set normalize parameters
normalize = gdata.vision.transforms.Normalize(
    [0.485,0.456,0.406], [0.229, 0.224, 0.225]
)
norm_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    normalize
])

# add semantic embeddings for each class label, those embeddings were provided by AwA2
file_name = '../data/AwA2/Animals_with_Attributes2/classes.txt'
def read_class_names(file_name):
    class_names = []
    with open(file_name, 'r') as ifile:
        for line in ifile:
            line_sp = line.rstrip('\n').split('\t')
            class_name = line_sp[-1]
            class_names.append(class_name)
    return class_names
class_names = read_class_names(file_name)

file_name = '../data/AwA2/Animals_with_Attributes2/predicate-matrix-continuous.txt'
def read_semantic_features(feature_file_name, class_names):
    semantic_features = {}
    with open(feature_file_name, 'r') as f_file:
        for i, line in enumerate(f_file):
            line_sp = line.rstrip('\n').split(' ')
            clean_line_sp = [float(x) for x in line_sp if len(x) > 0]
            semantic_features[class_names[i]] = np.asarray(clean_line_sp)
    return semantic_features

semantic_features = read_semantic_features(file_name, class_names)

# get pretrained resnet101 model
res101 = model_zoo.vision.resnet101_v2(pretrained=True)
res101_features = res101.features

# data preparation
train_iter = gdata.DataLoader(train_imgs.transform_first(norm_augs), batch_size=128, shuffle=True)
test_iter = gdata.DataLoader(test_imgs.transform_first(norm_augs), batch_size=128)

# images feature extraction
# it takes long long long time
train_data = []
train_class_features = [] # shape=(train_data.shape[0], 85)
for train_img in train_iter:
    train_data.extend(res101_features(train_img[0]).asnumpy())
    train_batch_labels = train_img[1].asnumpy()
    train_batch_features = [semantic_features[train_imgs.synsets[x]] for x in train_batch_labels]
    train_class_features.extend(train_batch_features)

train_data_arr = np.array(train_data)
train_data_arr = normalizeFeature(train_data_arr.transpose()).transpose()
train_class_features_arr = np.array(train_class_features)

test_labels = []
test_classes_id = np.arange(0, 10)
test_data = []
test_class_features = []
for test_img in test_iter:
    test_data.extend(res101_features(test_img[0]).asnumpy())
    test_batch_labels = test_img[1].asnumpy()
    test_labels.extend(test_batch_labels)

test_data_arr = np.array(test_data)

test_class_features = [semantic_features[x] for x in test_imgs.synsets] # shape=(10,85)
test_class_features_arr = np.array(test_class_features)

ld = 500000

W = SAE(train_data_arr.transpose(), train_class_features_arr.transpose(), ld)

semantic_predicted = np.dot(test_data_arr, normalizeFeature(W).transpose())
HITK = 1
zsl_accruacy, y_hit_k = zsl_acc(semantic_predicted, test_class_features_arr, test_classes_id, test_labels)