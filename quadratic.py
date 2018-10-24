#!/usr/bin/python
#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 載入數據集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每一個批次的大小
batch_size = 100

# 計算一共有多少批次
n_batch = mnist.train.num_examples // batch_size

# 定義兩個placeholder，目的在於 train時候透過 feed 傳入 x_data 與 y_data
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  # 用來 dropout 的機率
lr=tf.Variable(1e-3,dtype=tf.float32)#学习率为0.0001
# 建立一個神經網路

# 隱藏層1
W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]))
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_dropout = tf.nn.dropout(L1, keep_prob)

# 隱藏層2
W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000]))
L2 = tf.nn.tanh(tf.matmul(L1_dropout, W2) + b2)
L2_dropout = tf.nn.dropout(L2, keep_prob)

# 隱藏層3
W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000]))
L3 = tf.nn.tanh(tf.matmul(L2_dropout, W3) + b3)
L3_dropout = tf.nn.dropout(L3, keep_prob)

# 輸出層
W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))
prediction = tf.nn.tanh(tf.matmul(L3_dropout, W4) + b4)

# 代價函數 : loss = mean((y - prediction)^2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# gd = tf.train.GradientDescentOptimizer(0.2)
# 最小化 代價函數 (operator)
# train = gd.minimize(loss)
gd=tf.train.AdamOptimizer(lr)
train= gd.minimize(loss)
# 初始化變數 operator
init = tf.global_variables_initializer()

# 結果存在一個 boolean 的變數中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax 回傳一維張量中最大的值，所在的位置

# 求準確率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 開始training
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        sess.run(tf.assign(lr,0.0001*(0.95**epoch)))#学习率不断迭代，趋紧于0
        for batch in range(n_batch):  # 每一個 outer loop 疊代 n_batch 個批次

            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        # 計算一次準確率
        #train_feed_dict = {x: mnist.train.images, y: mnist.train.labels, keep_prob: 0.7}  # train data feed dictionary
        #train_acc = sess.run(accuracy, train_feed_dict)
        test_feed_dict = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 0.7}  # testing data feed dictionary
        test_acc = sess.run(accuracy, test_feed_dict)
        learning_rate=sess.run(lr)#计算学习率
        print "Iter " + str(epoch) +"  "+ "testing accuracy " + str(test_acc)+" learning rate"+str(learning_rate)

        #need to be optimize
        #from the result the cross entropy is faster and more accurate.

        #if dropout=0.7 only 70% neturon will work,and the speed will be lower
