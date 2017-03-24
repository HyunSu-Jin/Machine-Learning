import tensorflow as tf

x1_data = [73.,93.,89.,96.,73.]
x2_data  = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)


y_data = [152.,185.,180.,196.,142.]
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([1]),name="weight1")
W2 = tf.Variable(tf.random_normal([1]),name="weight2")
W3 = tf.Variable(tf.random_normal([1]),name="weight3")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = X1 * W1 + X2 * W2 + X3 * W3 +b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
	cost_val, hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={
		X1 : x1_data,
		X2: x2_data,
		X3 : x3_data,
		Y : y_data
		})
	if(step % 20 == 0):
		print(step,"Cost : ",cost_val,"\nPrediction :\n",hy_val)