import tensorflow as tf

x_data = [
	[0,0],
	[0,1],
	[1,0],
	[1,1]
]
y_data = [0,1,1,0]

X = tf.placeholder(tf.float32,shape=(None,2))
Y = tf.placeholder(tf.float32,shape=(None))

W = tf.Variable(tf.random_normal([2,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name = "bias")

logits = tf.matmul(X,W) +b
hypothesis = tf.sigmoid(logits)

cost = tf.reduce_mean(-Y * tf.log(hypothesis) - (1-Y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5 , dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(10001):
		cost_val, _ = sess.run([cost,optimizer],feed_dict={
			X : x_data,
			Y : y_data
			})
		if step % 100 ==0:
			print(step,cost_val)

	h,p,a = sess.run([hypothesis,predicted,accuracy],feed_dict={
		X : x_data,
		Y : y_data
		})
	print('hypothesis : ',h,"\npredicted : ",p,"\naccuracy : ",a)