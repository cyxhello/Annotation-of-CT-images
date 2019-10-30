import tensorflow as tf;
import numpy as np;
import os;
import sys;
import pandas as pd;
import matplotlib.pyplot as plt;
from skimage.io import imread;
from skimage.transform import resize;
from tqdm import tqdm;
from dataset import *;
from cnn import *;
from lstm import *;
from words import *;
from coco.coco import *;
from coco.pycocoevalcap.eval import *;

class Loader(object):
	def __init__(self,mean_file):
		self.rgb = True;
		self.scale_shape = np.array([224,224],np.int32);
		self.crop_shape = np.array([224,224],np.int32);
		self.mean = np.load(mean_file).mean(1).mean(1);

	def load(self,files):
		images = [];
		for image_file in files:
			print(image_file);
			image_file = './test/images/'+image_file;
			image = imread(image_file);
			if(self.rgb):
				temp = image.swapaxes(0,2);
				temp = temp[::-1];
				image = temp.swapaxes(0,2);
			image = resize(image,(self.scale_shape[0],self.scale_shape[1]));
			offset = (self.scale_shape-self.crop_shape)/2;
			offset = offset.astype(np.int32);
			image = image[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1],:];
			image = image-self.mean;
			images.append(image);
		images = np.array(images,np.float32);
		return images;

class Model(object):
	def __init__(self,params,phase):
		self.params = params;
		self.phase = phase;
		self.cnn_model = params.cnn;
		self.train_cnn = params.train_cnn;
		self.num_lstm = params.num_lstm;
		self.lstm_dim = params.lstm_dim;
		self.save_dir = os.path.join(os.path.join(params.save_dir,self.cnn_model+'/'),params.solver+'/');
		self.wordtable = WordTable(params.vocabulary_size,params.word_embed,params.sentence_length,params.word_file);
		self.wordtable.load();
		self.imageloader = Loader(params.mean_file);
		self.image_shape = [224,224,3];
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.global_step = tf.Variable(0,name='global_step',trainable=False);
		self.saver = tf.train.Saver(max_to_keep = 100);
		self.build();

	def build(self):
		print('Building the Model......');
		if(self.train_cnn):
			cnn = CNN(self.params,self.phase);
			lstm = LSTM(self.params,self.phase,self.wordtable,cnn.features);
		else:
			cnn = CNN(self.params,self.phase);
			lstm = LSTM(self.params,self.phase,self.wordtable);

		self.cnn = cnn;
		self.lstm = lstm;
		self.loss0 = lstm.loss0/tf.reduce_sum(lstm.masks);
		if self.train_cnn:
			self.loss1 = self.params.weight_decay*(tf.add_n(tf.get_collection('l2_0'))+tf.add_n(tf.get_collection('l2_1')));
		else:
			self.loss1 = self.params.weight_decay*tf.add_n(tf.get_collection('l2_1'));
		self.loss = self.loss0+self.loss1;
		
		if self.params.solver == 'adam':
			solver = tf.train.AdamOptimizer(self.params.learning_rate);
		elif self.params.solver == 'momentum':
			solver = tf.train.MomentumOptimizer(self.params.learning_rate,self.params.momentum);
		elif self.params.solver == 'rmsprop':
			solver = tf.train.RMSPropOptimizer(self.params.learning_rate,self.params.weight_decay,self.params.momentum);
		else:
			solver = tf.train.GradientDescentOptimizer(self.params.learning_rate);

		tensorflow_variables = tf.trainable_variables();
		gradients,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tensorflow_variables),3.0);
		optimizer = solver.apply_gradients(zip(gradients,tensorflow_variables),global_step=self.global_step);
		self.optimizer = optimizer;
		print('Model completed......');

	def train(self,sess,data,coco):
		print("Training the model.......");
		epochs = self.params.epochs;
		for epoch in tqdm(list(range(epochs)),desc='Epoch'):
			for i in tqdm(list(range(data.batches)),desc='Batch'):
				batch = data.next_batch();
				if(self.train_cnn):
					batch_data = self.feed(batch,train=True);
				else:
					files,_,_ = batch;
					images = self.imageloader.load(files);
					features = sess.run(self.cnn.features, feed_dict = {self.cnn.images:images, self.cnn.train:False});
					batch_data = self.feed(batch,train=True,featues = features);
				
				_, loss0, loss1, global_step = sess.run([self.optimizer,self.loss0,self.loss1,self.global_step],feed_dict=batch_data);
				print(" Loss0=%f Loss1=%f" %(loss0, loss1));
				if ((global_step+1)%self.params.save_period)==0:
					self.save(sess);
			data.reset();
		self.save(sess);
		print("Training completed");
	
	def val(self,sess,data,coco):
		print("Validating the model........");
		results = [];
		result_dir = self.params.val_result;
		for i in tqdm(list(range(data.count)),desc='Batch'):
			batch = data.next_batch();
			files = batch;
			files = files[0];
			image_name = os.path.splitext(files.split(os.sep)[-1])[0];
			if(self.train_cnn):
				batch_data = self.feed(batch,train=False);
			else:
				images = self.imageloader.load(files);
				features = sess.run(self.cnn.features, feed_dict = {self.cnn.images:images, self.cnn.train:False});
				batch_data = self.feed(batch,train=False,features=features);

			result = sess.run(self.lstm.results, feed_dict=batch_data);
			sentence = self.word_table.indices_to_sent(result.squeeze());
			results.append({'image_id': data.images[i], 'caption': sentence});
			img = imread(img_file);
			plt.imshow(img);
			plt.axis('off');
			plt.title(sentence);
			plt.savefig(os.path.join(result_dir, img_name+'_result.jpg'));
		data.reset();
		'''
		sent = pd.read_csv(self.params.val_file);
		results= [];
		for i in range(0,sent.shape[0]):
			results.append({'image_id':sent['image_id'][i],'caption':sent['caption'][i]});
		'''
		result_coco = coco.loadRes2(results);
		scorer = COCOEvalCap(coco, result_coco);
		scorer.evaluate();
		print("Validation completed");

	def test(self,sess,data):
		print("Testing the model........");
		results = [];
		result_dir = self.params.test_result;
		result_file = self.params.result_file;
		for i in tqdm(list(range(data.count)),desc='Batch'):
			batch = data.next_batch();
			image_files = batch;
			image_file = image_files[0];
			image_name = os.path.splitext(image_file.split(os.sep)[-1])[0];
			if(self.train_cnn):
				batch_data = self.feed(batch,train=False);
			else:
				images = self.imageloader.load(image_file);
				features = sess.run(self.cnn.features, feed_dict = {self.cnn.images:images, self.cnn.train:False});
				batch_data = self.feed(batch,train=False,features=features);

			result = sess.run(self.lstm.results, feed_dict=batch_data);
			sentence = self.wordtable.indices_to_sent(result.squeeze());
			results.append({'image_id': data.images[i], 'caption': sentence});
			image = imread(self.params.test_image+image_file);
			plt.imshow(image);
			plt.axis('off');
			plt.title(sentence);
			plt.savefig(os.path.join(result_dir, image_name+'_result.jpg'));
			
		data.reset();
		result = pd.DataFrame(results);
		result.to_csv(result_file)
		print("Testing completed");

	def load(self,sess):
		print("Loading model.....");
		checkpoint = tf.train.get_checkpoint_state(self.save_dir);
		if checkpoint is None:
			print("Error: No saved model found. Please train first.");
			sys.exit(0);
		self.saver.restore(sess, checkpoint.model_checkpoint_path);

	def load_cnn(self,sess,cnn_path):
		print("Loading CNN model from %s..." %data_path);
		data_dict = np.load(cnn_path).item();
		count = 0;
		miss_count = 0;
		for op_name in data_dict:
			with tf.variable_scope(op_name, reuse=True):
				for param_name, data in data_dict[op_name].iteritems():
					try:
						var = tf.get_variable(param_name);
						session.run(var.assign(data));
						count += 1;
					except ValueError:
						miss_count += 1;
						if not ignore_missing:
							raise
		print("%d variables loaded. %d variables missed." %(count, miss_count));

	def save(self, sess):
		print(("Saving model to %s" % self.save_dir));
		self.saver.save(sess,self.save_dir,self.global_step);

	def feed(self,batch,train,features = None):
		if(train):
			files,captions,masks = batch;
			images = self.imageloader.load(files);
			for i in range(self.batch_size):
				word_weight = self.lstm.word_weight[captions[i, :]];                
				masks[i, :] = masks[i, :]*word_weight;
				masks[i, :] = masks[i, :]*self.lstm.position_weight;
			if(self.train_cnn):
				return {self.cnn.images: images, self.lstm.captions: captions, self.lstm.masks: masks, self.lstm.train: train};
			else:
				return {self.lstm.features: features, self.lstm.captions: captions, self.lstm.masks: masks, self.lstm.train: train};
		else:
			files = batch;
			images = self.imageloader.load(files);
			captions = np.zeros((self.batch_size, self.params.sentence_length), np.int32);
			if(self.train_cnn):
				return {self.cnn.images: images, self.lstm.captions: captions, self.lstm.train: train};
			else:
				return {self.lstm.features: features, self.lstm.captions: captions, self.lstm.train: train};