import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# define 2 variables
batch_size = 50  # put 100 pics to train once

m_batch = mnist.train.num_examples

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
B = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + B)

loss = tf.reduce_mean(tf.square(y - prediction))#Quadratic Cost function
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))#cross entropy function tf.reduce.mean() means to get the average number of the function
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # bool changes into float32

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(m_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print "Iter" + str(epoch) + "testing accuracy" + str(acc)

        #need to be optimize
        #from the result the cross entropy is faster and more accurate.
