import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;

class CNN(object):
	def __init__(self,params,phase):
		self.cnn_model = params.cnn;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.batch_norm = params.batch_norm;
		if(self.cnn_model=='vgg16'):
			self.VGG16();

	def VGG16(self):
		print('Building the VGG-16 component......');
		image_shape = [224,224,3];
		bn = self.batch_norm;
		images = tf.placeholder(tf.float32,[self.batch_size]+image_shape);
		train = tf.placeholder(tf.bool);

		conv1_1 = convolution(images,3,3,64,1,1,'conv1_1');
		conv1_1 = batch_norm(conv1_1,'bn1_1',train,bn,'relu');
		conv1_2 = convolution(conv1_1,3,3,64,1,1,'conv1_2');
		conv1_2 = batch_norm(conv1_2,'bn1_2',train,bn,'relu');
		pool1 = max_pool(conv1_2,2,2,2,2,'pool1');

		conv2_1 = convolution(pool1,3,3,128,1,1,'conv2_1');
		conv2_1 = batch_norm(conv2_1,'bn2_1',train,bn,'relu');
		conv2_2 = convolution(conv2_1,3,3,128,1,1,'conv2_2');
		conv2_2 = batch_norm(conv2_2,'bn2_2',train,bn,'relu');
		pool2 = max_pool(conv2_2,2,2,2,2,'pool2');

		conv3_1 = convolution(pool2,3,3,256,1,1,'conv3_1');
		conv3_1 = batch_norm(conv3_1,'bn3_1',train,bn,'relu');
		conv3_2 = convolution(conv3_1,3,3,256,1,1,'conv3_2');
		conv3_2 = batch_norm(conv3_2,'bn3_2',train,bn,'relu');
		conv3_3 = convolution(conv3_2,3,3,256,1,1,'conv3_3');
		conv3_3 = batch_norm(conv3_3,'bn3_3',train,bn,'relu');
		pool3 = max_pool(conv3_3,2,2,2,2,'pool3');

		conv4_1 = convolution(pool3,3,3,512,1,1,'conv4_1');
		conv4_1 = batch_norm(conv4_1,'bn4_1',train,bn,'relu');
		conv4_2 = convolution(conv4_1,3,3,512,1,1,'conv4_2');
		conv4_2 = batch_norm(conv4_2,'bn4_2',train,bn,'relu');
		conv4_3 = convolution(conv4_2,3,3,512,1,1,'conv4_3');
		conv4_3 = batch_norm(conv4_3,'bn4_3',train,bn,'relu');
		pool4 = max_pool(conv4_3,2,2,2,2,'pool4');

		conv5_1 = convolution(pool4,3,3,512,1,1,'conv5_1');
		conv5_1 = batch_norm(conv5_1,'bn5_1',train,bn,'relu');
		conv5_2 = convolution(conv5_1,3,3,512,1,1,'conv5_2');
		conv5_2 = batch_norm(conv5_2,'bn5_2',train,bn,'relu');
		conv5_3 = convolution(conv5_2,3,3,512,1,1,'conv5_3')
		conv5_3 = batch_norm(conv5_3,'bn5_3',train,bn,'relu');

		conv5_3 = tf.reshape(conv5_3,[self.batch_size,196,512]);
		self.features = conv5_3;
		self.feature_shape = [196,512];
		self.images = images;
		self.train = train;
		print('VGG-16 built......');

	def basic_block1(self,feats,name1,name2,train,bn,c,s=2):
		'''A basic block of ResNets'''
		branch1 = convolution_no_bias(feats,1,1,4*c,s,s,name1+'_branch1');
		branch1 = batch_norm(branch1,name2+'_branch1',train,bn,None);
		branch2a = convolution_no_bias(feats,1,1,c,s,s,name1+'_branch2a');
		branch2a = batch_norm(branch2a,name2+'_branch2a',train,bn,'relu');
		branch2b = convolution_no_bias(branch2a,3,3,c,1,1,name1+'_branch2b');
		branch2b = batch_norm(branch2b,name2+'_branch2b',train,bn,'relu');
		branch2c = convolution_no_bias(branch2b,1,1,4*c,1,1,name1+'_branch2c');
		branch2c = batch_norm(branch2c,name2+'_branch2c',train,bn,None);
		output = branch1+branch2c;
		output = nonlinear(output, 'relu');
		return output;

	def basic_block2(self,feats,name1,name2,train,bn,c):
		'''Another basic block of ResNets'''
		branch2a = convolution_no_bias(feats,1,1,c,1,1,name1+'_branch2a');
		branch2a = batch_norm(branch2a,name2+'_branch2a',train,bn,'relu');
		branch2b = convolution_no_bias(branch2a,3,3,c,1,1,name1+'_branch2b');
		branch2b = batch_norm(branch2b,name2+'_branch2b',train,bn,'relu');
		branch2c = convolution_no_bias(branch2b,1,1,4*c,1,1,name1+'_branch2c');
		branch2c = batch_norm(branch2c,name2+'_branch2c',train,bn,None);
		output = feats+branch2c;
		output = nonlinear(output,'relu');
		return output;

