import os;
import math;
import numpy as np;
import tensorflow as tf;
from nn import *;
from words import *;

class LSTM(object):
	def __init__(self,params,phase,wordtable,features=None):
		self.num_lstm = params.num_lstm;
		self.lstm_dim = params.lstm_dim;
		self.word_embed = params.word_embed;
		self.decode_embed = params.decode_embed;
		self.init_layers = params.init_layers;
		self.sentence_length = params.sentence_length;
		self.batch_size = params.batch_size if phase=='train' else 1;
		self.batch_norm = params.batch_norm;
		self.wordtable = wordtable;
		self.params = params; 
		if(params.cnn=='vgg16'):
			self.feature_shape = [196,512];
		else:
			self.feature_shape = [49,2048];
		if(features is not None):
			self.features = features;
		else:
			self.features = tf.placeholder(tf.float32,[self.batch_size]+self.feature_shape);
		self.LSTM_Model();

	def LSTM_Model(self):
		print('Building the LSTM component......');
		train = tf.placeholder(tf.bool);
		bn = self.batch_norm;
		num_words = self.wordtable.num_words;
		captions = tf.placeholder(tf.int32,[self.batch_size,self.sentence_length]);
		masks = tf.placeholder(tf.float32,[self.batch_size,self.sentence_length]);
		self.word_weight = np.exp(-np.array(self.wordtable.word_freq));
		self.position_weight = np.exp(-np.array(list(range(self.sentence_length)))*0.003);
		idx2vec = np.array([self.wordtable.word2vec[self.wordtable.idx2word[i]] for i in range(num_words)]);
		emb_w = weight('emb_w', [num_words, self.word_embed], init_val=idx2vec, group_id=1);
		dec_w = weight('dec_w', [self.decode_embed, num_words], group_id=1);
		dec_b = bias('dec_b', [num_words], init_val=0.0);
		contexts = self.features;
		context_mean = tf.reduce_mean(contexts,1);
		init_features = context_mean;
		lstm = tf.contrib.rnn.LSTMCell(self.lstm_dim, initializer=tf.random_normal_initializer(stddev=0.03));
		if(self.num_lstm==1):
			temp = init_features;
			for i in range(self.init_layers):
				temp = fully_connected(temp,self.lstm_dim,'init_lstm_fc1'+str(i),group_id=1)
				temp = batch_norm(temp,'init_lstm_bn1'+str(i),train,bn,'tanh');
			memory = tf.identity(temp);
			temp = init_features;
			for i in range(self.init_layers):
				temp = fully_connected(temp,self.lstm_dim,'init_lstm_fc2'+str(i),group_id=1);
				temp = batch_norm(temp,'init_lstm_bn2'+str(i),train,bn,'tanh');
			output = tf.identity(temp);
			state = tf.contrib.rnn.LSTMStateTuple(memory,output);                   

		else:
			temp = init_features;
			for i in range(self.init_layers):
				temp = fully_connected(temp,self.lstm_dim,'init_lstm_fc11'+str(i),group_id=1);
				temp = batch_norm(temp,'init_lstm_bn11'+str(i),train,bn,'tanh');
			memory1 = tf.identity(temp);
			temp = init_features;
			for i in range(self.init_layers):
				temp = fully_connected(temp,self.lstm_dim,'init_lstm_fc12'+str(i),group_id=1);
				temp = batch_norm(temp,'init_lstm_bn12'+str(i),train,bn,'tanh');
			output1 = tf.identity(temp);
			temp = init_features;
			for i in range(self.init_layers):
				temp = fully_connected(temp,self.lstm_dim,'init_lstm_fc21'+str(i),group_id=1);
				temp = batch_norm(temp,'init_lstm_bn21'+str(i),train,bn,'tanh');
			memory2 = tf.identity(temp);
			temp = init_features;
			for i in range(self.init_layers):
				temp = fully_connected(temp,self.lstm_dim,'init_lstm_fc22'+str(i),group_id=1);
				temp = batch_norm(temp,'init_lstm_bn22'+str(i),train,bn,'tanh');
			output = tf.identity(temp);
			state1 = tf.contrib.rnn.LSTMStateTuple(memory1, output1)                
			state2 = tf.contrib.rnn.LSTMStateTuple(memory2, output)   

		loss0 = 0.0;
		results = [];
		scores = [];
		context_num = self.feature_shape[0];
		context_dim = self.feature_shape[1];
		context_flat = tf.reshape(contexts,[-1,context_dim]);
		with tf.variable_scope(tf.get_variable_scope()) as scope:
			for i in range(self.sentence_length):
				# Attention mechanism
				context_encode1 = fully_connected(context_flat,context_dim,'att_fc11',group_id=1); 
				context_encode1 = batch_norm(context_encode1,'att_bn11',train,bn,None);
				context_encode2 = fully_connected_no_bias(output,context_dim,'att_fc12',group_id=1);
				context_encode2 = batch_norm(context_encode2,'att_bn12',train,bn,None); 
				context_encode2 = tf.tile(tf.expand_dims(context_encode2,1),[1,context_num,1]);                 
				context_encode2 = tf.reshape(context_encode2,[-1, context_dim]);    
				context_encode = context_encode1 + context_encode2; 
				context_encode = nonlinear(context_encode, 'relu'); 
				context_encode = dropout(context_encode,0.5,train);
				alpha = fully_connected(context_encode, 1,'att_fc2',group_id=1)                 
				alpha = batch_norm(alpha,'att_bn2',train, bn,None)
				alpha = tf.reshape(alpha, [-1,context_num]);                                                           
				alpha = tf.nn.softmax(alpha); 
				if(i==0):   
					word_emb = tf.zeros([self.batch_size,self.word_embed]);
					weighted_context = tf.identity(context_mean);
				else:
					word_emb = tf.cond(train,lambda: tf.nn.embedding_lookup(emb_w, captions[:,i-1]),lambda: word_emb);
					weighted_context = tf.reduce_sum(contexts*tf.expand_dims(alpha,2),1);
				 
				if(self.num_lstm == 1):
					with tf.variable_scope("lstm"):
						output,state = lstm(tf.concat([weighted_context,word_emb],1),state);
				else:
					with tf.variable_scope("lstm1"):
						output1,state1 = lstm(weighted_context,state1);
					with tf.variable_scope("lstm2"):
						output,state2 = lstm(tf.concat([word_emb,output1],1),state2);
				
				expanded_output = tf.concat([output,weighted_context,word_emb],1);
				decoded = fully_connected(expanded_output,self.decode_embed,'dec_fc',group_id=1);
				decoded = nonlinear(decoded,'tanh');
				decoded = dropout(decoded,0.5,train);
				logits = tf.nn.xw_plus_b(decoded,dec_w,dec_b);
				# Update the loss
				cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=captions[:,i]);
				cross_entropy = cross_entropy*masks[:,i];
				loss0 += tf.reduce_sum(cross_entropy);
				max_prob_word = tf.argmax(logits,1);
				results.append(max_prob_word);
				probabilities = tf.nn.softmax(logits);
				score = tf.reduce_max(probabilities,1);
				scores.append(score);
				# Prepare for the next iteration
				word_emb = tf.cond(train, lambda: word_emb, lambda: tf.nn.embedding_lookup(emb_w,max_prob_word));         
				tf.get_variable_scope().reuse_variables();
		
		results = tf.stack(results,axis=1);
		scores = tf.stack(scores,axis=1);
		self.contexts = contexts;
		self.captions = captions;
		self.masks = masks;
		self.train = train;
		self.loss0 = loss0;
		self.results = results;
		self.scores = scores;
		print('LSTM component built......');