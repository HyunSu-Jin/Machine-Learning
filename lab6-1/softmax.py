import tensorflow as tf

#multinomial classifier
# x1,x2,x3,x4 >> a or b or c as one_hot
#soft max is needed

x_train=[[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6]]
y_train=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0]]

nb_classes = 3 # a,b,c

X  = tf.placeholder(tf.float32,shape=[None,4])
Y = tf.placeholder(tf.float32,shape=[None,3])

# tensorflow varialbes .. naming
W= tf.Variable(tf.random_normal([4,3]) , name="weight")
b = tf.Variable(tf.random_normal([3]) , name = "bias")

# matmul(X,W) is constant number

# hypothesis : the [0,0,0] one_hot structure, indicate the class
hypothesis = tf.nn.softmax(tf.matmul(X,W) +b)

# cost = 1/ N * sum of ( Y * log(-hypothesis))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis) , axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(2001):
		sess.run(train,feed_dict={
			X : x_train,
			Y : y_train
			})
		if step %200 == 0:
			print(step,sess.run(cost,feed_dict={
				X : x_train,
				Y : y_train
				}))
	# push x_data to Model and predict the class

	# testing & one_hot encoding
	a = sess.run(hypothesis,feed_dict={
		X : [[1,11,7,9],[1,3,4,3],[1,1,0,1]]
		})
	print(a,sess.run(tf.argmax(a,axis=1))) # return the maximum argument
