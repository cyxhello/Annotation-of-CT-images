from datetime import datetime
import math
import time
import tensorflow as tf

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        
        kernel = tf.get_variable(scope+"w", shape=[kh,kw,n_in,n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # conv2d对输入的tensor进行卷积处理
        # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
        # input是输入的图像tensor shape=[batch, in_height, in_width, in_channels]
        # [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
        # filter是卷积核tensor shape=[filter_height, filter_width, in_channels, out_channels]
        # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
        # strides卷积时在每一维上的步长，strides[0]=strides[3]=1
        # padding drop or zeropad
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME'
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation

# 定义 创建全连接层 函数fc_op
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
       
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        
        activation = tf.nn.relu_layer(input_op, kernel, biases, name= scope)
        p += [kernel, biases]
        return activation

# 定义 创建最大池化层 函数mpool_op
def mpool_op(input_op, name, kh, kw, dh, dw):
    
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def inference_op(input_op,keep_prob):
    p = []
   
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1,
                      dw=1, p=p)
    # 第一段卷积的第2个卷积层 卷积核3*3，共64个卷积核（输出通道数），步长1*1
    # input_op：224*224*64 输出尺寸224*224*64
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1,
                      dw=1, p=p)
    # 第一段卷积的pooling层，核2*2，步长2*2
    # input_op：224*224*64 输出尺寸112*112*64
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)

    # 第2段卷积的第一个卷积层 卷积核3*3，共128个卷积核（输出通道数），步长1*1
    # input_op：112*112*64 输出尺寸112*112*128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1,
                      dw=1, p=p)
    # input_op：112*112*128 输出尺寸112*112*128
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1,
                      dw=1, p=p)
    # input_op：112*112*128 输出尺寸56*56*128
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # 第3段卷积的第一个卷积层 卷积核3*3，共256个卷积核（输出通道数），步长1*1
    # input_op：56*56*128 输出尺寸56*56*256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1,
                      dw=1, p=p)
    # input_op：56*56*256 输出尺寸56*56*256
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1,
                      dw=1, p=p)
    # input_op：56*56*256 输出尺寸56*56*256
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1,
                      dw=1, p=p)
    # input_op：56*56*256 输出尺寸28*28*256
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # 第3段卷积的第一个卷积层 卷积核3*3，共512个卷积核（输出通道数），步长1*1
    # input_op：28*28*256 输出尺寸28*28*512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：28*28*512 输出尺寸28*28*512
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：28*28*512 输出尺寸28*28*512
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：28*28*512 输出尺寸14*14*512
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # 第5段卷积的第一个卷积层 卷积核3*3，共512个卷积核（输出通道数），步长1*1
    # input_op：14*14*512 输出尺寸14*14*512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：14*14*512 输出尺寸14*14*512
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：14*14*512 输出尺寸14*14*512
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：28*28*512 输出尺寸7*7*512
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)


    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    # tf.reshape(tensor, shape, name=None) 将tensor变换为参数shape的形式。
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


def run_benchmark():
    
    with tf.Graph().as_default():
        image_size = 224
        
        images = tf.Variable(tf.random_normal([batch_size,   
                                               image_size,
                                               image_size, 3],
                                               dtype = tf.float32,
                                               stddev=1e-1))  

        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)  

        
        init = tf.global_variables_initializer() 
        sess = tf.Session()  
        sess.run(init)

        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")  

        objective = tf.nn.l2_loss(fc8)  
        grad = tf.gradients(objective, p)  
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")  
batch_size = 32  
num_batches = 100
run_benchmark()
