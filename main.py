import sys;
import os;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';

import tensorflow as tf;
import argparse;
from model import *;
from dataset import *;
from coco.coco import *;

def main(argv):
	parser = argparse.ArgumentParser();
	'''Model Architecture'''
	parser.add_argument('--phase', default='train', help='Train,Validate or Test');
	parser.add_argument('--load', action='store_true', default=False, help='Load the trained model');
	parser.add_argument('--cnn', default='vgg16', help='It can be VGG16 or Resnet50 or Resnet101 or Resnet152');
	parser.add_argument('--load_cnn', action='store_true', default=False, help='Load the pretrained CNN model');
	parser.add_argument('--train_cnn', action='store_true', default=True, help='Train both CNN and LSTM. Otherwise, only LSTM is trained');
	parser.add_argument('--num_lstm', type=int, default=1, help='Number of LSTMs to use: Can be 1 or 2');
	parser.add_argument('--lstm_dim', type=int, default=1000, help='Hidden state dimension in each LSTM');
	parser.add_argument('--word_embed', type=int, default=300, help='Dimension of the word embedding')
	parser.add_argument('--decode_embed', type=int, default=1000, help='Dimension of the vector used for word generation');
	parser.add_argument('--init_layers', type=int, default=2, help='Number of layers to initialize the LSTMs')
	parser.add_argument('--sentence_length', type=int, default=30, help='Maximum Length of the generated caption');
	parser.add_argument('--vocabulary_size', type=int, default=5000, help='Maximum vocabulary size')

	'''Files and Directories'''
	parser.add_argument('--cnn_file',default='./cnn/VGG16.model', help='Trained model for CNN');
	parser.add_argument('--mean_file', default='./cnn/mean.npy', help= 'Dataset mean file for Image pre-processing');
	parser.add_argument('--word_file', default='./words/word_table.pickle', help='Word table file');
	parser.add_argument('--train_image', default='/home/arun/Projects/Tensorflow-Caption/train/images/', help='Directory containing the training images');
	parser.add_argument('--train_caption', default='/home/arun/Projects/Tensorflow-Caption/train/captions_train2014.json', help='Captions of the training images');
	parser.add_argument('--train_annotation', default='/home/arun/Projects/Tensorflow-Caption/train/anns.csv', help='Annotations file');
	parser.add_argument('--val_image', default='./val/images/', help='Directory containing the validation images');
	parser.add_argument('--val_caption', default='./val/captions_val2014.json', help='Captions of the validation images')
	parser.add_argument('--val_result', default='./val/results/', help='Directory to store the validation results as images');
	parser.add_argument('--val_file', default='./val/results.csv',help='File to store the validation results');
	parser.add_argument('--test_image', default='./test/images/', help='Directory containing the testing images');
	parser.add_argument('--result_file', default='./test/results.csv', help='File to store the testing results');
	parser.add_argument('--test_result', default='./test/results/', help='Directory to store the testing results as images');
	parser.add_argument('--save_dir', default='./models/', help='Directory to contain the trained model');
	parser.add_argument('--save_period', type=int, default=2000, help='Period to save the trained model');

	'''Hyper Parameters'''
	parser.add_argument('--solver', default='sgd', help='Gradient Descent Optimizer to use: Can be adam, momentum, rmsprop or sgd') 
	parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs');
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size');
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate');
	parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay');
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for some optimizers)'); 
	parser.add_argument('--decay', type=float, default=0.9, help='Decay (for some optimizers)'); 
	parser.add_argument('--batch_norm', action='store_true', default=False, help='Turn on to use batch normalization');

	args = parser.parse_args();
	with tf.Session() as sess:
		if(args.phase=='train'):
			coco,data = train_data(args);
			model = Model(args,'train');
			sess.run(tf.global_variables_initializer());
			if(args.load):
				model.load(sess);
			elif(args.load_cnn):
				model.load_cnn(sess,args.cnn_file);
			model.train(sess,data,coco);
		elif(args.phase=='val'):
			coco,data = val_data(args);
			model = Model(args,'val');
			sess.run(tf.global_variables_initializer());
			model.load(sess);
			model.val(sess,data,coco);
		else:
			data = test_data(args);
			model = Model(args,'test');
			sess.run(tf.global_variables_initializer());
			model.load(sess);
			model.test(sess,data);

main(sys.argv);