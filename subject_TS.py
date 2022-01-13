#-*- coding: utf-8 -*-
from subject_input import return_train_set, return_test_set
import tensorflow as tf
import numpy as np
import time
import os.path

# 超参数
epoch_num = 100 # 0.016 min/epoch
learning_rate = 0.0001
learning_rate_decay = 0.997
use_exponential_decay = True # 学习率是否使用指数衰减
regularize_l2_w = 0.01 # 正则化系数
channels = 32 # 电极(通道)数量
window_size = 128 # 滑动窗口
n_classes = 2  # 分类
batch_size = 30  # 每批次样本数
label_index = 0 # label_index: [0,1,2,3]-[valence,arousal,dominance,liking]

# 训练
sub = 0 # 第几个人[0,31]
train_steps = 1*40*42 // batch_size # 计算训练集一共有多少个批次
test_steps = 1*40*18 // batch_size # 计算测试集一共有多少个批次
print_step = train_steps // 2 # 每隔多少步打印训练损失
method = 2 # 0,1,2,3,4-----0-原始，1-均值，2-均值升级，3-筛选1s，4-筛选2s
file_path = os.path.abspath(os.path.dirname(os.getcwd())) + "\\Database\\baseline_removed_database"

# 定义卷积操作
def conv1d_op(inputs, name, kernel_width, out_channels, strides, regularize, padding="SAME"):
    # num_in是输入的深度，这个参数被用来确定过滤器的输入通道数
    in_channels = inputs.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kernel_width, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection("regularizer_loss", regularize(kernel))
        conv = tf.nn.conv1d(inputs, kernel, strides, padding=padding)
        #biases = tf.Variable(tf.constant(0.1, shape=[out_channels], dtype=tf.float32), trainable=True, name="b")
        # 在训练时，需要将第二个参数training = True; 在测试时，将training = False
        bn = tf.layers.batch_normalization(conv, training=True)
        return tf.nn.relu(bn, name=scope)

# 定义全连操作
def fc_op(inputs, name, num_out, regularize, activation=True):
    # num_in为输入单元的数量
    num_in = inputs.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope + "w", shape=[num_in, num_out], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection("regularizer_loss", regularize(weights))
        biases = tf.Variable(tf.constant(0.1, shape=[num_out], dtype=tf.float32), name="b")
        if activation: return tf.nn.relu(tf.matmul(inputs, weights) + biases)
        else: return tf.matmul(inputs, weights) + biases

# 占位符
x = tf.placeholder(tf.float32, shape=[None, 32, window_size], name='x')
y = tf.placeholder(tf.int64, shape=[None], name='y')

# L2正则化
regularize_l2 = tf.contrib.layers.l2_regularizer(regularize_l2_w)

# 时间卷积 TS, 32x64 # shape=[None(batch), 32, window_size]
conv2_1 = conv1d_op(x, name="conv2_1", kernel_width=1, out_channels=256, strides=1, regularize=regularize_l2)
conv2_2 = conv1d_op(conv2_1, name="conv2_2", kernel_width=1, out_channels=512, strides=1, regularize=regularize_l2)
conv2_3 = conv1d_op(conv2_2, name="conv2_3", kernel_width=1, out_channels=1024, strides=1, regularize=regularize_l2)
conv2_4 = conv1d_op(conv2_3, name="conv2_4", kernel_width=1, out_channels=64, strides=1, regularize=regularize_l2)

# 结果汇总为一个向量的形式
shape2 = conv2_4.get_shape().as_list()
flattened_shape2 = shape2[1] * shape2[2]
reshaped2 = tf.reshape(conv2_4, [-1, flattened_shape2], name="reshaped2") # 时间卷积

# 全连接层1
fc1_1 = fc_op(reshaped2, name="fc1_1", num_out=512, regularize=regularize_l2)
# 全连接层2
fc1_2 = fc_op(fc1_1, name="fc1_2", num_out=512, regularize=regularize_l2)
# 全连接层3
result = fc_op(fc1_2, name="result", num_out=n_classes, regularize=regularize_l2, activation=False)

# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=y))
loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection("regularizer_loss"))

# 学习率指数衰减, 使用AdamOptimizer进行优化
training_step = tf.Variable(0, trainable=False)
if use_exponential_decay:
    learning_rate = tf.train.exponential_decay(learning_rate, training_step, train_steps, learning_rate_decay)
# batch_normalization 缓解DNN训练中的梯度消失问题 加快模型的训练速度
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=training_step)
""""""
# 求准确率
correct_prediction = tf.equal(tf.argmax(result,1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

fold_list = []  # 记录10折结果
fold_loss = []  # 记录10折损失
fold_time_strat = time.time()
for fold in range(1, 11, 2):  # 10折交叉验证
    # 加载数据集
    train_handle = return_train_set(file_path, batch_size, window_size, label_index, method, fold, sub)
    train_batch, labels_train = next(train_handle)
    test_handle = return_test_set(file_path, batch_size, window_size, label_index, method, fold, sub)
    test_batch, labels_test = next(test_handle)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        test_acc_list = []  # 测试精度存储列表
        train_loss_list = []  # 测试损失存储列表
        for epoch in range(epoch_num):
            print("#" * 70, "fold=%d" % fold)
            print("**epoch %d: " % (epoch + 1))
            epoch_start_time = time.time()

            for batch in range(train_steps):
                start_time = time.time()
                _, loss_value, train_accuracy = sess.run([train_op, loss, accuracy],
                                                         feed_dict={x: train_batch, y: labels_train})
                if (batch + 1) % print_step == 0:
                    duration = time.time() - start_time
                    print("batch %d: loss = %.3f, train_accuracy = %.3f%% (%.3f sec/batch)"
                          % (batch, loss_value, train_accuracy * 100, duration))
                train_batch, labels_train = next(train_handle)
                epoch_loss = loss_value

            epoch_loss_value = "%.3f" % (epoch_loss)  # 保留小数点后3位
            train_loss_list.append(float(epoch_loss_value)) #每一个epoch的损失列表

            true_count = 0
            total_sample_count = test_steps * batch_size  # 统计用于测试的帧数
            for test_step in range(test_steps):
                predictions = sess.run([correct_prediction], feed_dict={x: test_batch, y: labels_test})
                true_count += np.sum(predictions)
                test_batch, labels_test = next(test_handle)

            sec_pre_epoch = time.time() - epoch_start_time
            
            test_acc = "%.3f" % (true_count / total_sample_count * 100)  # 保留小数点后3位
            test_acc_list.append(float(test_acc)) #每一个epoch的精度列表
           
            print("total_test_samples = %d, true_samples = %d" % (total_sample_count, true_count))
            print("%.3f min/epoch, acc_test =" % (sec_pre_epoch / 60), test_acc_list)

    fold_list.append(test_acc_list) #每一折的精度列表
    fold_loss.append(train_loss_list) #每一折的精度列表

fold_time = time.time() - fold_time_strat
print("#" * 70, "Sub %d Done!" % sub)
print("10fold_time = %.3f min" % (fold_time / 60))

acc = np.array(fold_list).reshape([5, epoch_num])  # shape=[10,epoch_num]
loss_end = np.array(fold_loss).reshape([5, epoch_num])  # shape=[10,epoch_num]
#np.save("record\method%d_label%d_total.npy" % (method, label_index), acc)
print("10fold_acc =", np.mean(acc, axis=0))
print("10fold_loss =", np.mean(loss_end, axis=0))
#np.save("record\method%d_label%d_mean.npy" % (method, label_index), np.mean(acc, axis=0))

""""""
