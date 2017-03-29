import tensorflow as tf
import numpy as np

xy= np.loadtxt('data-04-zoo.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_classes = 7
# X,Y,W,b,hypothesis,cost

X = tf.placeholder(tf.float32,shape=[None,16])
Y =tf.placeholder(tf.int32,shape=[None,1])
# transformation : Y to one_hot
Y_one_hot = tf.one_hot(Y,nb_classes) # [ [0],[1]] > [[[10000000]], [[0100000]]]
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes])

W = tf.Variable(tf.random_normal([16,nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

logits = tf.matmul(X,W) + b # z = WX
hypothesis = tf.nn.softmax(logits) # predict

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#prediction, accuracy
prediction = tf.argmax(hypothesis,1) # the index of Y ...predict
correct_prediction = tf.equal(prediction,tf.argmax(Y_one_hot,1)) #1 or 0
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(2001):
		sess.run(optimizer,feed_dict={
			X : x_data,
			Y : y_data
			})
		if step % 100 == 0:
			loss, acc = sess.run([cost,accuracy],feed_dict={
				X : x_data,
				Y : y_data
				})
			print("Step : {:5}\t Loss : {:.3f}\t Acc : {:.2f}".format(step,loss,acc))
	pred = sess.run(prediction,feed_dict={
		X : x_data
		})
	for p,y in zip(pred,y_data.flatten()):
		print("[{}] Prediction : {} True Y : {}".format(p==int(y),p,int(y)))