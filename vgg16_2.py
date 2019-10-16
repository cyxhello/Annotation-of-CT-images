from datetime import datetime
import math
import time
import tensorflow as tf

# VGG包含很多卷积，函数conv_op创建卷积层并把本层参数存入参数列表
# input_op是输入的tensor
# name是本层名称
# kh卷积核高
# kw卷积核宽
# n_out是卷积核的数量，即输出通道数
# dh步长的高
# dw步长的宽
# p是参数列表
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    # get_shape获得输入tensor的通道数
    # 224*224*3的图片就是最后的3
    n_in = input_op.get_shape()[-1].value
    # name_scope将scope内生成的variable自动命名为name/xxx
    # 用于区分不同卷积层的组件
    with tf.name_scope(name) as scope:
        # 卷积核参数由get_variable函数创建
        # shape即卷积核的高，宽，输入通道数，输出通道数
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
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        # bias使用tf.constant赋值为0
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        # tf.variable再将其转换成可训练的参数
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        # tf.nn.bias_add将卷积结果和bias相加
        z = tf.nn.bias_add(conv,biases)
        # 使用relu对卷积结果进行非线性处理
        activation = tf.nn.relu(z, name=scope)
        # 把卷积层使用到的参数添加到参数列表p
        p += [kernel, biases]
        return activation

# 定义 创建全连接层 函数fc_op
def fc_op(input_op, name, n_out, p):
    # 同样获得输入图片tensor的通道数
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # 同样使用get_variable创建全连接层的参数，只不过参数纬度只有两个:输入和输出通道数
        # 参数初始化方法使用xavier_initializer
        kernel = tf.get_variable(scope+"w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # bias利用constant函数初始化为较小的值0.1,而不是0
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        # 这里使用tf.nn.relu_layer，对输入变量input_op和kernel做矩阵乘法并加上biases
        # 再做relu非线性变换得到activation
        activation = tf.nn.relu_layer(input_op, kernel, biases, name= scope)
        # 把全连接层使用到的参数添加到参数列表p
        p += [kernel, biases]
        return activation

# 定义 创建最大池化层 函数mpool_op
def mpool_op(input_op, name, kh, kw, dh, dw):
    # 直接使用tf.nn.max_pool,输入为图片tensor，池化尺寸为kh*kw,步长为dh*dw，padding为same
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

# 完成了 卷积层，全连接层，pooling层 的创建函数
# 下面开始创建VGG16的网络结构


# inference_op是创建网络结构的函数
# input_op是输入的图像tensor shape=[batch, in_height, in_width, in_channels]
# keep_prob是控制dropout比率的一个placeholder
def inference_op(input_op,keep_prob):
    # 初始化参数p列表
    p = []
    # VGG16包含6个部分，前面5段卷积，最后一段全连接
    # 每段卷积包含多个卷积层和pooling层

    # 下面是第一段卷积，包含2个卷积层和一个pooling层
    # 利用前面定义好的函数conv_op,mpool_op 创建这些层
    # 第一段卷积的第一个卷积层 卷积核3*3，共64个卷积核（输出通道数），步长1*1
    # input_op：224*224*3 输出尺寸224*224*64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1,
                      dw=1, p=p)
    # 第一段卷积的第2个卷积层 卷积核3*3，共64个卷积核（输出通道数），步长1*1
    # input_op：224*224*64 输出尺寸224*224*64
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1,
                      dw=1, p=p)
    # 第一段卷积的pooling层，核2*2，步长2*2
    # input_op：224*224*64 输出尺寸112*112*64
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)

    # 下面是第2段卷积，包含2个卷积层和一个pooling层
    # 第2段卷积的第一个卷积层 卷积核3*3，共128个卷积核（输出通道数），步长1*1
    # input_op：112*112*64 输出尺寸112*112*128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1,
                      dw=1, p=p)
    # input_op：112*112*128 输出尺寸112*112*128
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1,
                      dw=1, p=p)
    # input_op：112*112*128 输出尺寸56*56*128
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # 下面是第3段卷积，包含3个卷积层和一个pooling层
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

    # 下面是第4段卷积，包含3个卷积层和一个pooling层
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

    # 前面4段卷积发现，VGG16每段卷积都是把图像面积变为1/4，但是通道数翻倍
    # 因此图像tensor的总尺寸缩小一半

    # 下面是第5段卷积，包含3个卷积层和一个pooling层
    # 第3段卷积的第一个卷积层 卷积核3*3，共512个卷积核（输出通道数），步长1*1
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

    # 将第五段卷积网络的结果扁平化
    # reshape将每张图片变为7*7*512=25088的一维向量
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    # tf.reshape(tensor, shape, name=None) 将tensor变换为参数shape的形式。
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # 第一个全连接层，是一个隐藏节点数为4096的全连接层
    # 后面接一个dropout层，训练时保留率为0.5，预测时为1.0
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    # 第2个全连接层，是一个隐藏节点数为4096的全连接层
    # 后面接一个dropout层，训练时保留率为0.5，预测时为1.0
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    # 最后是一个1000个输出节点的全连接层
    # 利用softmax输出分类概率
    # argmax输出概率最大的类别
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p

# 评测VGGNet每轮计算时间的函数
# session Tensorflow的session
# target需要评测的运算算子
def time_tensorflow_run(session, target, feed, info_string): # 与AlexNet非常相似，session参数一点点区别
    # num_steps_burn_in预热轮数，前几轮有显存加载，可以跳过时间评测
    # 只计算10轮迭代后的计算时间
    num_steps_burn_in = 10
    # 总时间
    total_duration = 0.0
    # 总时间的平方
    total_duration_squared = 0.0
    # 进行num_batches + num_steps_burn_in次迭代计算
    for i in range(num_batches + num_steps_burn_in):
        # 使用time.time()记录时间
        start_time = time.time()
        # 每次迭代通过session.run执行
        _ = session.run(target, feed_dict=feed) # 引入feed_dict方便后面传入keep_prob来控制Dropout层的保留比率
        duration = time.time() - start_time
        # 在预热之后
        if i >= num_steps_burn_in:
            # 每10轮迭代显示当前迭代所需时间
            if not i % 10:
                print( '%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    # 计算每轮迭代的平均耗时和标准差
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    # 并不使用imagenet数据集训练，而是使用随机图片数据测试前奎和反馈耗时
    with tf.Graph().as_default():
        image_size = 224
        # 利用随机数生成之前的图片tensor
        images = tf.Variable(tf.random_normal([batch_size,   # 生成随机的图片224*224
                                               image_size,
                                               image_size, 3],
                                               dtype = tf.float32,
                                               stddev=1e-1))  # 标准差为0.1的正态分布的随机数

        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)  # 构建网络结构获得参数列表

        # 通过tf.session创建新的session，并通过global_variable初始化所有参数
        # 并通过run运行该session 只有运行run并feed数据运算才真正执行
        init = tf.global_variables_initializer()  # 初始化全局参数
        sess = tf.Session()  # 创建session
        sess.run(init)

        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")  # 预测时节点保留率

        objective = tf.nn.l2_loss(fc8)  # 计算VGGNet-16最后的全连接层的输出fc8的L2 loss
        grad = tf.gradients(objective, p)  # 使用tf.gradients求相对于这个loss的所有模型参数的梯度
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")  # 这里的target为求解梯度的操作grad

batch_size = 32  # VGGNet-16模型的体积较大，如果使用较大的batch_size，GPU显存会不够用
num_batches = 100
run_benchmark()