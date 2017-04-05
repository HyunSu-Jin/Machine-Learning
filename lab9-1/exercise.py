import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10

X = tf.placeholder(tf.float32,shape=(None,784))
Y = tf.placeholder(tf.float32,shape=(None,nb_classes))

W = tf.Variable(tf.random_normal([784,nb_classes]),name="weight")
b = tf.Variable(tf.random_normal([nb_classes]),name="bias")

logits = tf.matmul(X,W) +b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis),axis=-1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.arg_max(hypothesis,1)
is_correct = tf.cast(tf.equal(prediction,tf.arg_max(Y,1)),dtype=tf.float32)
accuracy = tf.reduce_mean(is_correct)

training_epochs = 15
batch_size =100

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples / batch_size)

		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			cost_val, _ = sess.run([cost,train],feed_dict={
				X:batch_xs,
				Y :batch_ys
				})
			avg_cost += cost_val / total_batch

		print("Epoch : ",epoch+1,"\nCost :",avg_cost)

	accuracy = sess.run(accuracy,feed_dict={
		X : mnist.test.images,
		Y : mnist.test.labels
		})
	print("Accuracy : ",accuracy)